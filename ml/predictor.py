"""
ml/predictor.py
===============
High-level prediction orchestrator.

Given a datetime, lat/lon, and historical lag data, this module:
  1. Builds feature vectors for each model
  2. Runs inference
  3. Returns predicted environmental parameters

Handles both:
  - Location-specific models (dict keyed by location name)
  - Global models (single model)

For location-specific models, selects the nearest reference location.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ml.model_loader import ModelBundle, ModelLoader
from utils.feature_builder import (
    build_precipitation_features,
    build_sunload_features,
    build_temperature_features,
    build_wind_speed_features,
)

logger = logging.getLogger(__name__)


class Predictor:
    """
    Run ML predictions for environmental parameters.

    Usage::

        loader = ModelLoader()
        loader.load_all()

        predictor = Predictor(loader)
        result = predictor.predict_all(
            dt=datetime(2026, 6, 15, 14, 0),
            lat=17.44, lon=78.50,
            historical_lags={...},
        )
    """

    def __init__(self, model_loader: ModelLoader) -> None:
        self._loader = model_loader

    def predict_all(
        self,
        dt: datetime,
        lat: float,
        lon: float,
        historical_lags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run all available models and return predicted environmental values.

        Parameters
        ----------
        dt : datetime
            Target prediction datetime.
        lat, lon : float
            Location coordinates.
        historical_lags : dict | None
            Output from WeatherService.get_historical_lags():
                temps, precip, wind, sunload, wind_dir

        Returns
        -------
        dict with keys:
            temperature_c, precipitation_class, precipitation_label,
            wind_speed_kmh, solar_radiation_wm2, models_used
        """
        lags = historical_lags or {}
        results: Dict[str, Any] = {}
        models_used: List[str] = []

        # ── Temperature ────────────────────────────────────────
        temp = self._predict_temperature(dt, lat, lon, lags.get("temps", []))
        if temp is not None:
            results["temperature_c"] = round(temp, 2)
            models_used.append("temperature")
        else:
            results["temperature_c"] = None

        # ── Wind Speed ─────────────────────────────────────────
        wind = self._predict_wind_speed(
            dt, lat, lon,
            wind_direction=lags.get("wind_dir", 180.0),
            historical_wind=lags.get("wind", []),
        )
        if wind is not None:
            results["wind_speed_kmh"] = round(wind, 2)
            models_used.append("wind_speed")
        else:
            results["wind_speed_kmh"] = None

        # ── Sunload ────────────────────────────────────────────
        sunload = self._predict_sunload(
            dt, lat, lon, lags.get("sunload", []),
        )
        if sunload is not None:
            results["solar_radiation_wm2"] = round(sunload, 2)
            models_used.append("sunload")
        else:
            results["solar_radiation_wm2"] = None

        # ── Precipitation ──────────────────────────────────────
        precip = self._predict_precipitation(
            dt, lat, lon,
            temperature=results.get("temperature_c", 25.0) or 25.0,
            wind_speed=results.get("wind_speed_kmh", 5.0) or 5.0,
            solar_radiation=results.get("solar_radiation_wm2", 200.0) or 200.0,
            historical_precip=lags.get("precip", []),
        )
        if precip is not None:
            results["precipitation_class"] = int(precip)
            class_map = self._get_precip_class_map()
            results["precipitation_label"] = class_map.get(int(precip), "Unknown")
            models_used.append("precipitation")
        else:
            results["precipitation_class"] = None
            results["precipitation_label"] = None

        results["models_used"] = models_used
        logger.info("ML prediction complete — models used: %s", models_used)
        return results

    # ─────────────────────────────────────────────────────────────
    # Temperature
    # ─────────────────────────────────────────────────────────────
    def _predict_temperature(
        self, dt: datetime, lat: float, lon: float,
        historical_temps: List[float],
    ) -> Optional[float]:
        bundle = self._loader.get("temperature")
        if bundle is None:
            logger.warning("Temperature model not loaded, skipping.")
            return None

        features = build_temperature_features(dt, lat, lon, historical_temps)
        return self._run_model(bundle, features)

    # ─────────────────────────────────────────────────────────────
    # Wind Speed
    # ─────────────────────────────────────────────────────────────
    def _predict_wind_speed(
        self, dt: datetime, lat: float, lon: float,
        wind_direction: float,
        historical_wind: List[float],
    ) -> Optional[float]:
        bundle = self._loader.get("wind_speed")
        if bundle is None:
            logger.warning("Wind speed model not loaded, skipping.")
            return None

        features = build_wind_speed_features(dt, lat, lon, wind_direction, historical_wind)
        return self._run_model(bundle, features, lat=lat, lon=lon)

    # ─────────────────────────────────────────────────────────────
    # Sunload
    # ─────────────────────────────────────────────────────────────
    def _predict_sunload(
        self, dt: datetime, lat: float, lon: float,
        historical_sunload: List[float],
    ) -> Optional[float]:
        bundle = self._loader.get("sunload")
        if bundle is None:
            logger.warning("Sunload model not loaded, skipping.")
            return None

        features = build_sunload_features(dt, lat, lon, historical_sunload)
        return self._run_model(bundle, features, lat=lat, lon=lon)

    # ─────────────────────────────────────────────────────────────
    # Precipitation
    # ─────────────────────────────────────────────────────────────
    def _predict_precipitation(
        self, dt: datetime, lat: float, lon: float,
        temperature: float, wind_speed: float, solar_radiation: float,
        historical_precip: List[float],
    ) -> Optional[int]:
        bundle = self._loader.get("precipitation")
        if bundle is None:
            logger.warning("Precipitation model not loaded, skipping.")
            return None

        features = build_precipitation_features(
            dt, lat, lon,
            temperature=temperature,
            wind_speed=wind_speed,
            solar_radiation=solar_radiation,
            historical_precip=historical_precip,
        )
        result = self._run_model(bundle, features, lat=lat, lon=lon)
        return int(result) if result is not None else None

    # ─────────────────────────────────────────────────────────────
    # Generic model runner
    # ─────────────────────────────────────────────────────────────
    def _run_model(
        self,
        bundle: ModelBundle,
        features: Dict[str, float],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        quantile: str = "P50",
    ) -> Optional[float]:
        """
        Build feature array in correct order and run inference.

        Handles:
          - Single model (bundle.model is a model object)
          - Location-dict (bundle.model is a dict of location → model)
          - Quantile-dict (model is a dict like {"P10": m, "P50": m, "P90": m})

        For quantile models, defaults to "P50" (median prediction).
        """
        try:
            # Build ordered feature array
            feature_array = np.array(
                [features[fname] for fname in bundle.feature_names],
                dtype=np.float64,
            ).reshape(1, -1)

            model = bundle.model

            # If model is a dict, resolve to a single model object
            if isinstance(model, dict):
                # Check if keys are quantile labels (P10, P50, P90)
                keys = set(model.keys())
                if keys & {"P10", "P50", "P90"}:
                    # Quantile model — select requested quantile
                    model = model.get(quantile, model.get("P50", next(iter(model.values()))))
                    logger.info("Using quantile %s model for %s", quantile, bundle.name)
                else:
                    # Location-keyed dict — pick nearest location's model
                    model = self._select_nearest_model(
                        model, lat or 0.0, lon or 0.0, bundle.metadata,
                    )
                    # Nested quantile check after location selection
                    if isinstance(model, dict) and (set(model.keys()) & {"P10", "P50", "P90"}):
                        model = model.get(quantile, model.get("P50", next(iter(model.values()))))

            # Run prediction
            if hasattr(model, "predict"):
                pred = model.predict(feature_array)
                return float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            else:
                logger.error(
                    "Model %s has no predict() method (type: %s)",
                    bundle.name, type(model),
                )
                return None

        except KeyError as exc:
            logger.error(
                "Feature mismatch for %s — missing key %s. "
                "Expected features: %s, got: %s",
                bundle.name, exc, bundle.feature_names, list(features.keys()),
            )
            return None
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", bundle.name, exc, exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────
    # Location model selection
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _select_nearest_model(
        model_dict: Dict[str, Any],
        lat: float,
        lon: float,
        metadata: Dict[str, Any],
    ) -> Any:
        """
        Given a dict mapping location names to models, select the
        model for the nearest reference location.
        """
        locations = metadata.get("locations", {})
        if not locations:
            # No location metadata — return first model
            return next(iter(model_dict.values()))

        best_key = None
        best_dist = float("inf")

        for loc_name, coords in locations.items():
            rlat = coords.get("lat", 0)
            rlon = coords.get("lon", 0)
            dist = math.sqrt((lat - rlat) ** 2 + (lon - rlon) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_key = loc_name

        if best_key and best_key in model_dict:
            logger.info("Selected model for location '%s' (dist=%.2f°)", best_key, best_dist)
            return model_dict[best_key]

        # Fallback — try case-insensitive match
        for key in model_dict:
            if best_key and key.lower() == best_key.lower():
                return model_dict[key]

        # Last resort
        return next(iter(model_dict.values()))

    def _get_precip_class_map(self) -> Dict[int, str]:
        """Return precipitation class mapping from model metadata."""
        bundle = self._loader.get("precipitation")
        if bundle and "class_mapping" in bundle.metadata:
            return bundle.metadata["class_mapping"]
        return {0: "No Rain", 1: "Low Rain", 2: "Moderate Rain", 3: "High Rain"}
