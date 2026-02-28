# SUMO Scenario Generation API

**Production-ready REST API** that generates SUMO-compatible scenario files from geographic coordinates and datetime.

Combines real-time weather data (Open-Meteo), elevation data, solar radiation calculations, and ML model predictions to produce environment-aware SUMO simulation configurations.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POST /generate-scenario                         │
│                  { lat, lon, date, time }                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────▼────────┐
              │ Decision Engine │──── date within 10 days? ──┐
              └───────┬────────┘                             │
                      │ NO                                   │ YES
          ┌───────────▼───────────┐               ┌──────────▼──────────┐
          │   ML Model Pipeline   │               │  Open-Meteo Forecast│
          │ ┌───────────────────┐ │               │  (direct API call)  │
          │ │ Historical Lags   │ │               └──────────┬──────────┘
          │ │ Feature Builder   │ │                          │
          │ │ XGBoost/LightGBM  │ │                          │
          │ └───────────────────┘ │                          │
          │  Predicts:            │                          │
          │  • Temperature        │                          │
          │  • Wind Speed         │                          │
          │  • Solar Radiation    │                          │
          │  • Precipitation      │                          │
          └───────────┬───────────┘                          │
                      │                                      │
              ┌───────▼──────────────────────────────────────▼───────┐
              │              Environment Features Merge              │
              │  + Elevation (Open-Elevation API)                    │
              │  + Sun Load (astronomical model)                     │
              │  + Road Context (OSM / Overpass API)                 │
              └──────────────────────┬───────────────────────────────┘
                                     │
              ┌──────────────────────▼───────────────────────────────┐
              │              SUMO XML Generator                      │
              │  • weather.add.xml     (friction, visibility, wind)  │
              │  • environment.xml     (full env specification)      │
              │  • vehicle_types.xml   (speed/behaviour adjustments) │
              │  • traffic_lights.xml  (clearance time adjustments)  │
              │  • scenario_config.xml (metadata)                    │
              └─────────────────────────────────────────────────────┘
```

---

## ML Models

| Model | Framework | Features | Output |
|-------|-----------|----------|--------|
| **Temperature** | XGBoost | 7 (DOY sin/cos, lag-1/7/30, lat, lon) | Regression (°C) |
| **Precipitation** | LightGBM/XGBoost | 16 (date, T2M, wind, solar, lags, rolling sums) | 4-class (No/Low/Moderate/High Rain) |
| **Sunload** | XGBoost | 11 (lat, lon, hour/month cyclical, lag features) | Regression (W/m²) |
| **Wind Speed** | XGBoost | 13 (lat, lon, hour/month cyclical, direction, lags) | Regression (km/h) |

Models are auto-loaded from `mlmodels/MODELS/` at startup.

---

## Quick Start

### 1. Install dependencies

```bash
cd sumoapi
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 3. Start the server

```bash
# Development (with hot reload)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or simply:
python main.py
```

### 4. Generate a scenario

```bash
# Forecast mode (within 10 days)
curl -X POST http://localhost:8000/generate-scenario \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 17.4399,
    "longitude": 78.4983,
    "date": "2026-03-05",
    "time": "14:30"
  }'

# ML prediction mode (beyond forecast window)
curl -X POST http://localhost:8000/generate-scenario \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 17.4399,
    "longitude": 78.4983,
    "date": "2026-07-15",
    "time": "10:00"
  }'
```

### 5. Check API health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
```

### 6. Interactive docs

Open **http://localhost:8000/docs** for Swagger UI.

---

## Project Structure

```
sumoapi/
├── main.py                        # FastAPI entrypoint
├── .env                           # Environment configuration
├── requirements.txt               # Python dependencies
│
├── services/
│   ├── weather_service.py         # Open-Meteo weather data (async)
│   ├── elevation_service.py       # Terrain elevation lookup
│   ├── sun_service.py             # Solar radiation / sun load
│   └── osm_service.py             # OpenStreetMap road context
│
├── ml/
│   ├── model_loader.py            # Auto-detect & load ML models
│   └── predictor.py               # Feature building + inference
│
├── sumo/
│   └── xml_generator.py           # SUMO XML file generation
│
├── utils/
│   ├── config.py                  # Centralised settings (pydantic)
│   └── feature_builder.py         # ML feature vector construction
│
├── mlmodels/MODELS/               # Trained ML model pickles
│   ├── Temperature/
│   ├── Precipitation/
│   ├── Sunload/
│   └── Wind Speed/
│
└── output/sumo/                   # Generated SUMO scenario files
```

---

## API Reference

### `POST /generate-scenario`

**Request:**
```json
{
    "latitude": 17.4399,
    "longitude": 78.4983,
    "date": "2026-03-15",
    "time": "14:30"
}
```

**Response:**
```json
{
    "scenario_id": "scenario_20260315_1430_a1b2c3d4",
    "environment_features": {
        "temperature_c": 32.5,
        "rainfall_mm_1h": 0.0,
        "wind_speed_kmh": 12.3,
        "visibility_m": 24000,
        "surface_pressure_hpa": 1013.2,
        "cloud_cover_pct": 25,
        "solar_radiation_wm2": 780.5,
        "sun_load_factor": 0.7805,
        "elevation_m": 505.0,
        "weather_condition": "partly_cloudy",
        "road_condition": "dry",
        "is_day": true
    },
    "prediction_source": "forecast",
    "sumo_files_generated": [
        "output/sumo/scenario_.../weather.add.xml",
        "output/sumo/scenario_.../environment.xml",
        "output/sumo/scenario_.../vehicle_types.add.xml",
        "output/sumo/scenario_.../scenario_config.xml"
    ],
    "models_used": []
}
```

### `GET /health`

Returns API health status and loaded model count.

### `GET /models`

Returns detailed information about all loaded ML models.

---

## External APIs Used

| API | Purpose | Key Required |
|-----|---------|-------------|
| [Open-Meteo](https://open-meteo.com) | Weather forecast & historical data | ❌ Free |
| [Open-Elevation](https://open-elevation.com) | Terrain altitude | ❌ Free |
| [Overpass (OSM)](https://overpass-api.de) | Road network context | ❌ Free |

---

## SUMO Output Files

| File | Purpose |
|------|---------|
| `weather.add.xml` | Weather conditions: friction, visibility, precipitation |
| `environment.xml` | Full environmental specification with all parameters |
| `vehicle_types.add.xml` | Weather-adjusted vehicle behaviour (speed, sigma, minGap) |
| `traffic_lights.add.xml` | Extended clearance intervals for poor weather |
| `scenario_config.xml` | Scenario metadata and file manifest |

---

## Intelligence Features

- **Automatic forecast vs ML routing** — decides based on date delta
- **Response caching** — TTL-based cache for API responses
- **Graceful degradation** — if an API/model fails, uses fallbacks
- **Location-aware models** — selects nearest reference location
- **Climatological fallbacks** — seasonal estimates when no historical data
- **Structured logging** — full request tracing

---

## License

Internal — BluFleet AI™
