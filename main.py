"""
main.py
=======
FastAPI entrypoint for the SUMO Scenario Generation API.

Endpoints
---------
  POST /generate-scenario   — Generate SUMO scenario from lat/lon/date/time
  GET  /health               — Health check
  GET        # ── Fetch elevation ────────────────────────────────────────
        elevation = await e        # ── Build 1km x 1km SUMO road network from OSM ────────────
        logger.info(
            "Building 1km×1km network — bbox S=%.6f W=%.6f N=%.6f E=%.6f",
            bbox["south"], bbox["west"], bbox["north"], bbox["east"],
        )
        net_path = await build_network(
            lat=req.latitude,
            lon=req.longitude,
            output_dir=Path(get_settings().sumo_output_path) / scenario_id,
            elevation_m=elevation,
        )
        if net_path:
            logger.info("Network ready: %s (%.1f KB)", net_path, net_path.stat().st_size / 1024)
        else:
            logger.warning("Network build failed — scenario continues without .net.xml")

        # ── Generate SUMO XML files ────────────────────────────────
        files = xml_gen.generate(
            environment=environment,
            scenario_id=scenario_id,
            lat=req.latitude,
            lon=req.longitude,
            dt=target_dt,
            prediction_source=prediction_source,
            road_context=road_context,
            net_path=net_path,
        )vc.get_elevation(req.latitude, req.longitude)
        environment["elevation_m"] = elevation

        # ── Compute 1km x 1km bounding box ────────────────────────
        bbox = compute_bbox(req.latitude, req.longitude)
        environment["bbox"] = bbox

        # ── Fetch road context (non-blocking) ──────────────────────
        road_context = await osm_svc.get_road_context(req.latitude, req.longitude)els               — List loaded ML models

Usage
-----
    # Development
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

Example curl
------------
    curl -X POST http://localhost:8000/generate-scenario \\
      -H "Content-Type: application/json" \\
      -d '{
        "latitude": 17.4399,
        "longitude": 78.4983,
        "date": "2026-03-15",
        "time": "14:30"
      }'
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ml.model_loader import ModelLoader
from ml.predictor import Predictor
from services.elevation_service import ElevationService
from services.osm_service import OSMService, build_network, compute_bbox, fetch_road_context
from services.sun_service import SunService
from services.weather_service import WeatherService
from sumo.xml_generator import SUMOXMLGenerator
from utils.config import get_settings

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sumoapi")

# ──────────────────────────────────────────────────────────────────
# Singletons (initialised at startup)
# ──────────────────────────────────────────────────────────────────
weather_svc: WeatherService
elevation_svc: ElevationService
sun_svc: SunService
osm_svc: OSMService
model_loader: ModelLoader
predictor: Predictor
xml_gen: SUMOXMLGenerator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    global weather_svc, elevation_svc, sun_svc, osm_svc
    global model_loader, predictor, xml_gen

    logger.info("🚀 Starting SUMO Scenario Generation API …")

    # Initialise services
    weather_svc = WeatherService()
    elevation_svc = ElevationService()
    sun_svc = SunService()
    osm_svc = OSMService()
    xml_gen = SUMOXMLGenerator()

    # Load ML models
    model_loader = ModelLoader()
    model_loader.load_all()
    predictor = Predictor(model_loader)

    logger.info(
        "✅ API ready — %d ML models loaded: %s",
        len(model_loader.available_models),
        model_loader.available_models,
    )
    yield

    logger.info("🛑 Shutting down SUMO Scenario API.")


app = FastAPI(
    title="SUMO Scenario Generation API",
    description=(
        "Generate SUMO-compatible scenario files from geographic coordinates "
        "and datetime.  Fetches real-time weather data from Open-Meteo and "
        "uses ML models for beyond-forecast predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    """Input payload for scenario generation."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (decimal degrees)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (decimal degrees)")
    date: str = Field(..., description="Target date in YYYY-MM-DD format")
    time: str = Field(..., description="Target time in HH:MM format")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")
        return v

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("time must be in HH:MM format")
        return v


class EnvironmentFeatures(BaseModel):
    """Environmental parameters returned in the response."""
    temperature_c: Optional[float] = None
    rainfall_mm_1h: Optional[float] = None
    snowfall_cm: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    visibility_m: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    surface_pressure_hpa: Optional[float] = None
    cloud_cover_pct: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    sun_load_factor: Optional[float] = None
    elevation_m: Optional[float] = None
    weather_condition: Optional[str] = None
    road_condition: Optional[str] = None
    precipitation_label: Optional[str] = None
    is_day: Optional[bool] = None


class ScenarioResponse(BaseModel):
    """Response payload after scenario generation."""
    scenario_id: str
    environment_features: Dict[str, Any]
    prediction_source: str
    sumo_files_generated: List[str]
    models_used: List[str] = []
    bbox_1km: Dict[str, float] = {}
    network_file: Optional[str] = None
    run_command: str = ""


# ──────────────────────────────────────────────────────────────────
# POST /generate-scenario
# ──────────────────────────────────────────────────────────────────
@app.post("/generate-scenario", response_model=ScenarioResponse)
async def generate_scenario(req: ScenarioRequest):
    """
    Generate a complete SUMO scenario from location + datetime.

    **Flow:**
    1. Parse target datetime
    2. Decide forecast vs ML prediction
    3. Fetch weather + elevation + solar data
    4. If beyond forecast window → run ML models
    5. Generate SUMO XML files
    6. Return environment features + file paths
    """
    cfg = get_settings()

    # ── Parse datetime ─────────────────────────────────────────
    try:
        target_dt = datetime.strptime(f"{req.date} {req.time}", "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: {exc}")

    today = date.today()
    target_date = target_dt.date()
    days_ahead = (target_date - today).days

    # ── Decide prediction source ───────────────────────────────
    if 0 <= days_ahead <= cfg.forecast_window_days:
        prediction_source = "forecast"
    else:
        prediction_source = "ml_model"

    logger.info(
        "📍 Request: lat=%.4f lon=%.4f datetime=%s → source=%s (Δ%d days)",
        req.latitude, req.longitude, target_dt.isoformat(),
        prediction_source, days_ahead,
    )

    scenario_id = f"scenario_{target_dt.strftime('%Y%m%d_%H%M')}_{uuid.uuid4().hex[:8]}"
    environment: Dict[str, Any] = {}
    models_used: List[str] = []

    try:
        # ── Fetch elevation ────────────────────────────────────
        elevation = await elevation_svc.get_elevation(req.latitude, req.longitude)
        environment["elevation_m"] = elevation

        # ── Compute 1km x 1km bounding box ────────────────────
        bbox = compute_bbox(req.latitude, req.longitude)
        environment["bbox"] = bbox

        # ── Fetch road context (point query) ──────────────────
        road_context = await fetch_road_context(req.latitude, req.longitude)
        environment.update({k: v for k, v in road_context.items()
                            if k not in ("road_types", "speed_limits")})

        if prediction_source == "forecast":
            # ── FORECAST PATH ──────────────────────────────────
            snapshot = await weather_svc.get_snapshot_at(
                req.latitude, req.longitude, target_dt,
            )
            if snapshot:
                environment.update({
                    "temperature_c": snapshot.get("temperature_c"),
                    "rainfall_mm_1h": snapshot.get("rainfall_mm_1h"),
                    "snowfall_cm": snapshot.get("snowfall_cm"),
                    "wind_speed_kmh": snapshot.get("wind_speed_kmh"),
                    "wind_direction_deg": snapshot.get("wind_direction_deg"),
                    "visibility_m": snapshot.get("visibility_m"),
                    "relative_humidity_pct": snapshot.get("relative_humidity_pct"),
                    "surface_pressure_hpa": snapshot.get("surface_pressure_hpa"),
                    "cloud_cover_pct": snapshot.get("cloud_cover_pct"),
                    "weather_condition": snapshot.get("weather_condition"),
                    "road_condition": snapshot.get("road_condition"),
                    "is_day": snapshot.get("is_day"),
                    "shortwave_radiation": snapshot.get("shortwave_radiation"),
                })

                # Sun load calculation
                sun_data = sun_svc.compute_sun_load(
                    snapshot, req.latitude, req.longitude, target_dt,
                )
                environment.update({
                    "solar_radiation_wm2": sun_data["shortwave_radiation_wm2"],
                    "sun_load_factor": sun_data["sun_load_factor"],
                    "is_day": sun_data["is_day"],
                })
            else:
                logger.warning("No forecast snapshot available — falling back to ML")
                prediction_source = "ml_model"

        if prediction_source == "ml_model":
            # ── ML MODEL PATH ─────────────────────────────────
            # Get historical data for lag features
            lags = await weather_svc.get_historical_lags(
                req.latitude, req.longitude, target_date,
            )

            # Run ML predictions
            ml_results = predictor.predict_all(
                target_dt, req.latitude, req.longitude, lags,
            )
            environment.update({
                "temperature_c": ml_results.get("temperature_c"),
                "wind_speed_kmh": ml_results.get("wind_speed_kmh"),
                "solar_radiation_wm2": ml_results.get("solar_radiation_wm2"),
                "precipitation_class": ml_results.get("precipitation_class"),
                "precipitation_label": ml_results.get("precipitation_label"),
            })
            models_used = ml_results.get("models_used", [])

            # Sun load from astronomical model
            sun_data = sun_svc.compute_sun_load(
                None, req.latitude, req.longitude, target_dt,
            )
            environment.update({
                "sun_load_factor": sun_data["sun_load_factor"],
                "is_day": sun_data["is_day"],
            })

            # Derive road condition from precipitation class
            precip_class = ml_results.get("precipitation_class")
            if precip_class is not None:
                if precip_class == 0:
                    environment["road_condition"] = "dry"
                elif precip_class == 1:
                    environment["road_condition"] = "wet"
                elif precip_class == 2:
                    environment["road_condition"] = "wet"
                elif precip_class == 3:
                    environment["road_condition"] = "flooded"
            else:
                environment["road_condition"] = "dry"

            if environment.get("weather_condition") is None:
                environment["weather_condition"] = "clear_sky"

        # ── Build 1km x 1km SUMO network from OSM ────────────
        cfg = get_settings()
        net_path = await build_network(
            lat=req.latitude,
            lon=req.longitude,
            output_dir=Path(cfg.sumo_output_path) / scenario_id,
            elevation_m=elevation,
        )

        # ── Generate all SUMO XML files ────────────────────────
        files = xml_gen.generate(
            environment=environment,
            scenario_id=scenario_id,
            lat=req.latitude,
            lon=req.longitude,
            dt=target_dt,
            prediction_source=prediction_source,
            road_context=road_context,
            net_path=net_path,
        )

        logger.info("✅ Scenario %s generated — %d files", scenario_id, len(files))

        # Build the ready-to-run sumo-gui command
        sumocfg = next((f for f in files if f.endswith(".sumocfg")), None)
        run_cmd = f"sumo-gui -c {sumocfg}" if sumocfg else ""

        return ScenarioResponse(
            scenario_id=scenario_id,
            environment_features=environment,
            prediction_source=prediction_source,
            sumo_files_generated=files,
            models_used=models_used,
            bbox_1km=bbox,
            network_file=str(net_path) if net_path else None,
            run_command=run_cmd,
        )

    except Exception as exc:
        logger.error("❌ Scenario generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {str(exc)}")


# ──────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_loader.available_models,
        "models_count": len(model_loader.available_models),
    }


# ──────────────────────────────────────────────────────────────────
# GET /models
# ──────────────────────────────────────────────────────────────────
@app.get("/models")
async def list_models():
    """List all loaded ML models with metadata."""
    models_info = {}
    for name in model_loader.available_models:
        bundle = model_loader.get(name)
        if bundle:
            models_info[name] = {
                "model_type": bundle.model_type,
                "feature_count": len(bundle.feature_names),
                "feature_names": bundle.feature_names,
                "metadata_keys": list(bundle.metadata.keys()),
            }
    return {"models": models_info}


# ──────────────────────────────────────────────────────────────────
# Run with: python main.py
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.host,
        port=cfg.port,
        reload=cfg.debug,
        log_level=cfg.log_level.lower(),
    )
