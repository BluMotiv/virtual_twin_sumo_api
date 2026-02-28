# weather-api

Standalone weather data extraction module for the **BluFleet AI** virtual twin pipeline.

Fetches real-world weather data from **Open-Meteo** — completely free, no API key required.

---

## Folder Structure

```
weather-api/
├── fetch_weather.py          ← main entry point (run this)
├── src/
│   └── weather/
│       ├── __init__.py
│       └── open_meteo_client.py   ← API client + parser
├── data/
│   ├── raw/                  ← saved JSON weather snapshots
│   └── processed/            ← cleaned data (future use)
├── tests/
│   └── test_open_meteo.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quickstart

```bash
cd "BluFleet AI/weather-api"
pip install -r requirements.txt

# By place name
python fetch_weather.py --place "Hyderabad"

# By lat/lon
python fetch_weather.py --lat 17.4399 --lon 78.4983

# Historical range
python fetch_weather.py --place "Hyderabad" --start 2026-01-01 --end 2026-01-31 --save

# 3-day forecast
python fetch_weather.py --place "Hyderabad" --days 3 --save
```

---

## SUMO Parameters Extracted

| Parameter | Unit | Used For |
|---|---|---|
| `temperature_c` | °C | Road friction model (SA-2) |
| `rainfall_mm_1h` | mm | Road friction + visibility |
| `rainfall_mm_24h` | mm (rolling) | Road friction model (SA-2) |
| `snowfall_cm` | cm | Surface condition |
| `wind_speed_kmh` | km/h | Vehicle lateral stability |
| `wind_direction_deg` | ° | Directional wind effect |
| `visibility_m` | m | Sensor occlusion model |
| `relative_humidity_pct` | % | Condensation / fog |
| `surface_pressure_hpa` | hPa | Air density / aero drag |
| `cloud_cover_pct` | % | Lighting / solar irradiance |
| `weather_condition` | WMO string | SUMO scenario condition |
| `road_condition` | dry/wet/snowy/flooded | SUMO friction input |
| `is_day` | bool | SUMO lighting flag |

---

## API Used

| API | Key Required | Docs |
|---|---|---|
| **Open-Meteo Forecast** | ❌ None | https://open-meteo.com/en/docs |
| **Open-Meteo Historical** | ❌ None | https://open-meteo.com/en/docs/historical-weather-api |
| **Open-Meteo Geocoding** | ❌ None | https://open-meteo.com/en/docs/geocoding-api |

---

## Run Tests

```bash
python -m pytest tests/ -v
```
