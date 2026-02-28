"""
ml/model_loader.py
==================
Auto-detects, loads, and validates ML models from the mlmodels/MODELS
directory structure.

Directory layout expected:
    MODELS/
    ├── Temperature/
    │   ├── temperature_xgb_model.pkl       — XGBoost model
    │   ├── temperature_model_features.pkl  — feature name list
    │   └── temperature_model_package.pkl   — optional pipeline/scaler
    ├── Precipitation/
    │   ├── precipitation_models.pkl        — LightGBM / XGBoost dict
    │   ├── precipitation_features.pkl      — feature name list
    │   └── precipitation_metadata.pkl      — class mapping, thresholds
    ├── Sunload/
    │   ├── sunload_models.pkl              — model dict
    │   ├── features_sunload.pkl            — feature name list
    │   └── locations.pkl                   — reference location coords
    └── Wind Speed/
        ├── wind_models.pkl                 — model dict
        ├── features.pkl                    — feature name list
        └── locations.pkl                   — reference location coords

Supports: XGBoost, LightGBM, scikit-learn models.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

from utils.config import get_settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data classes for loaded model bundles
# ──────────────────────────────────────────────────────────────────
class ModelBundle:
    """Container for a loaded model, its features, and metadata."""

    def __init__(
        self,
        name: str,
        model: Any,
        feature_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        model_type: str = "unknown",
    ):
        self.name = name
        self.model = model
        self.feature_names = feature_names
        self.metadata = metadata or {}
        self.model_type = model_type

    def __repr__(self) -> str:
        return (
            f"ModelBundle(name={self.name!r}, type={self.model_type!r}, "
            f"features={len(self.feature_names)})"
        )


# ──────────────────────────────────────────────────────────────────
# Model Loader
# ──────────────────────────────────────────────────────────────────
class ModelLoader:
    """
    Discover, load, and cache ML models from disk.

    Usage::

        loader = ModelLoader()
        loader.load_all()

        bundle = loader.get("temperature")
        prediction = bundle.model.predict(feature_array)
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelBundle] = {}
        self._models_dir = get_settings().ml_models_path

    @property
    def available_models(self) -> List[str]:
        """Return names of successfully loaded models."""
        return list(self._models.keys())

    def get(self, name: str) -> Optional[ModelBundle]:
        """Get a loaded model bundle by name (case-insensitive)."""
        return self._models.get(name.lower())

    # ─────────────────────────────────────────────────────────────
    # Load all models
    # ─────────────────────────────────────────────────────────────
    def load_all(self) -> Dict[str, ModelBundle]:
        """Discover and load all model bundles from the models directory."""
        logger.info("Loading ML models from: %s", self._models_dir)

        if not self._models_dir.exists():
            logger.error("Models directory does not exist: %s", self._models_dir)
            return self._models

        self._load_temperature()
        self._load_precipitation()
        self._load_sunload()
        self._load_wind_speed()

        logger.info(
            "Loaded %d models: %s",
            len(self._models),
            list(self._models.keys()),
        )
        return self._models

    # ─────────────────────────────────────────────────────────────
    # Temperature
    # ─────────────────────────────────────────────────────────────
    def _load_temperature(self) -> None:
        folder = self._models_dir / "Temperature"
        if not folder.exists():
            logger.warning("Temperature model folder missing: %s", folder)
            return

        try:
            features = self._load_pickle(folder / "temperature_model_features.pkl")
            model_obj = self._load_pickle(folder / "temperature_xgb_model.pkl")

            # The pkl may be a raw model or a package dict
            model = model_obj
            metadata: Dict[str, Any] = {}
            if isinstance(model_obj, dict):
                model = model_obj.get("model", model_obj)
                metadata = {k: v for k, v in model_obj.items() if k != "model"}

            # Also try loading the package file for scaler info
            pkg_path = folder / "temperature_model_package.pkl"
            if pkg_path.exists():
                try:
                    pkg = self._load_pickle(pkg_path)
                    if isinstance(pkg, dict):
                        if "model" in pkg:
                            model = pkg["model"]
                        metadata.update({k: v for k, v in pkg.items() if k != "model"})
                except Exception:
                    pass

            model_type = self._detect_model_type(model)
            bundle = ModelBundle(
                name="temperature",
                model=model,
                feature_names=features,
                metadata=metadata,
                model_type=model_type,
            )
            self._models["temperature"] = bundle
            logger.info("✅ Temperature model loaded: %s", bundle)

        except Exception as exc:
            logger.error("❌ Failed to load temperature model: %s", exc)

    # ─────────────────────────────────────────────────────────────
    # Precipitation
    # ─────────────────────────────────────────────────────────────
    def _load_precipitation(self) -> None:
        folder = self._models_dir / "Precipitation"
        if not folder.exists():
            logger.warning("Precipitation model folder missing: %s", folder)
            return

        try:
            features = self._load_pickle(folder / "precipitation_features.pkl")
            metadata = self._load_pickle(folder / "precipitation_metadata.pkl")
            models_obj = self._load_pickle(folder / "precipitation_models.pkl")

            # models_obj is a dict mapping location name → classifier
            # e.g. {"Thrissur": XGBClassifier, "Kanpur": LGBMClassifier, ...}
            model = models_obj
            model_type = "ensemble"

            if isinstance(models_obj, dict):
                first_key = next(iter(models_obj))
                sample = models_obj[first_key]
                model_type = self._detect_model_type(sample)

                # Build location coordinate mapping for nearest-location selection
                # Use thresholds metadata to derive location names
                if isinstance(metadata, dict):
                    thresholds = metadata.get("percentile_thresholds_per_location", {})
                    # Map known training locations to approximate coordinates
                    _PRECIP_LOCS = {
                        "Thrissur":  {"lat": 10.5, "lon": 76.5},
                        "Kanpur":    {"lat": 26.5, "lon": 80.5},
                        "Leh":       {"lat": 34.0, "lon": 77.0},
                        "Jaisalmer": {"lat": 27.5, "lon": 71.5},
                        "Hyderabad": {"lat": 17.5, "lon": 78.5},
                        "Shillong":  {"lat": 25.5, "lon": 91.5},
                        "Araku":     {"lat": 18.3, "lon": 82.9},
                    }
                    metadata["locations"] = _PRECIP_LOCS
            else:
                model_type = self._detect_model_type(models_obj)

            bundle = ModelBundle(
                name="precipitation",
                model=model,
                feature_names=features,
                metadata=metadata if isinstance(metadata, dict) else {},
                model_type=model_type,
            )
            self._models["precipitation"] = bundle
            logger.info("✅ Precipitation model loaded: %s", bundle)

        except Exception as exc:
            logger.error("❌ Failed to load precipitation model: %s", exc)

    # ─────────────────────────────────────────────────────────────
    # Sunload
    # ─────────────────────────────────────────────────────────────
    def _load_sunload(self) -> None:
        folder = self._models_dir / "Sunload"
        if not folder.exists():
            logger.warning("Sunload model folder missing: %s", folder)
            return

        try:
            features = self._load_pickle(folder / "features_sunload.pkl")
            models_obj = self._load_pickle(folder / "sunload_models.pkl")

            metadata: Dict[str, Any] = {}
            locations_path = folder / "locations.pkl"
            if locations_path.exists():
                metadata["locations"] = self._load_pickle(locations_path)

            model = models_obj
            if isinstance(models_obj, dict):
                first_key = next(iter(models_obj))
                model_type = self._detect_model_type(models_obj[first_key])
            else:
                model_type = self._detect_model_type(models_obj)

            bundle = ModelBundle(
                name="sunload",
                model=model,
                feature_names=features,
                metadata=metadata,
                model_type=model_type,
            )
            self._models["sunload"] = bundle
            logger.info("✅ Sunload model loaded: %s", bundle)

        except Exception as exc:
            logger.error("❌ Failed to load sunload model: %s", exc)

    # ─────────────────────────────────────────────────────────────
    # Wind Speed
    # ─────────────────────────────────────────────────────────────
    def _load_wind_speed(self) -> None:
        folder = self._models_dir / "Wind Speed"
        if not folder.exists():
            logger.warning("Wind Speed model folder missing: %s", folder)
            return

        try:
            features = self._load_pickle(folder / "features.pkl")
            models_obj = self._load_pickle(folder / "wind_models.pkl")

            metadata: Dict[str, Any] = {}
            locations_path = folder / "locations.pkl"
            if locations_path.exists():
                metadata["locations"] = self._load_pickle(locations_path)

            model = models_obj
            if isinstance(models_obj, dict):
                first_key = next(iter(models_obj))
                model_type = self._detect_model_type(models_obj[first_key])
            else:
                model_type = self._detect_model_type(models_obj)

            bundle = ModelBundle(
                name="wind_speed",
                model=model,
                feature_names=features,
                metadata=metadata,
                model_type=model_type,
            )
            self._models["wind_speed"] = bundle
            logger.info("✅ Wind Speed model loaded: %s", bundle)

        except Exception as exc:
            logger.error("❌ Failed to load wind speed model: %s", exc)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Safely load a pickle file, falling back to joblib if needed."""
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            if _HAS_JOBLIB:
                logger.info("Pickle failed for %s, trying joblib…", path.name)
                return joblib.load(path)
            raise

    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """Auto-detect the ML framework from the model object type."""
        cls_name = type(model).__name__
        module = type(model).__module__ or ""

        if "xgboost" in module.lower() or "XGB" in cls_name:
            return "xgboost"
        if "lightgbm" in module.lower() or "LGBM" in cls_name or "Booster" in cls_name:
            return "lightgbm"
        if "sklearn" in module.lower():
            return "sklearn"
        if "torch" in module.lower():
            return "pytorch"
        if "tensorflow" in module.lower() or "keras" in module.lower():
            return "tensorflow"
        return f"unknown({cls_name})"
