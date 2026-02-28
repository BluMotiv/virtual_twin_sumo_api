"""
open_meteo_client.py
====================
Fetches real weather data from Open-Meteo (https://open-meteo.com).

✅  NO API KEY required.
✅  Completely free, no sign-up.
✅  Returns all parameters needed for a SUMO scenario.

Endpoints used
--------------
  Geocoding  : https://geocoding-api.open-meteo.com/v1/search
  Forecast   : https://api.open-meteo.com/v1/forecast
  Historical : https://archive-api.open-meteo.com/v1/archive

SUMO-relevant parameters returned
----------------------------------
  temperature_c          → road friction model (SA-2)
  rainfall_mm_1h         → road friction + visibility
  rainfall_mm_24h        → road friction model (SA-2)
  snowfall_cm            → surface condition
  snow_depth_m           → surface condition
  wind_speed_kmh         → vehicle drag / lateral stability
  wind_direction_deg     → directional wind effect
  visibility_m           → sensor occlusion model
  relative_humidity_pct  → condensation / fog model
  surface_pressure_hpa   → air density (affects aero drag)
  cloud_cover_pct        → lighting / solar irradiance
  weather_code           → WMO code (maps to SUMO condition string)
  is_day                 → lighting flag for SUMO
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------
_GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

# WMO weather code → human-readable condition
WMO_CODES: Dict[int, str] = {
    0:  "clear_sky",
    1:  "mainly_clear", 2: "partly_cloudy", 3: "overcast",
    45: "fog", 48: "rime_fog",
    51: "light_drizzle", 53: "moderate_drizzle", 55: "dense_drizzle",
    61: "slight_rain",   63: "moderate_rain",    65: "heavy_rain",
    71: "slight_snow",   73: "moderate_snow",    75: "heavy_snow",
    77: "snow_grains",
    80: "slight_showers", 81: "moderate_showers", 82: "violent_showers",
    85: "slight_snow_showers", 86: "heavy_snow_showers",
    95: "thunderstorm",
    96: "thunderstorm_hail", 99: "thunderstorm_heavy_hail",
}

# ---------------------------------------------------------------------------
# Hourly variables requested from Open-Meteo
# ---------------------------------------------------------------------------
_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "visibility",
    "is_day",
]


class OpenMeteoClient:
    """Fetch SUMO-ready weather data from Open-Meteo. No API key needed."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Geocoding (place name → lat/lon)
    # ------------------------------------------------------------------
    def geocode(self, place_name: str) -> Dict[str, Any]:
        """Resolve a place name to coordinates using Open-Meteo geocoding."""
        params = {"name": place_name, "count": 1, "language": "en", "format": "json"}
        resp = self.session.get(_GEO_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"Open-Meteo geocoder: no results for '{place_name}'.")
        r = results[0]
        logger.info(
            "Geocoded '%s' → %s (lat=%.4f, lon=%.4f, elev=%.0fm)",
            place_name, r.get("name"), r["latitude"], r["longitude"],
            r.get("elevation", 0),
        )
        return {
            "name":      r.get("name", place_name),
            "country":   r.get("country", ""),
            "lat":       r["latitude"],
            "lon":       r["longitude"],
            "elevation": r.get("elevation", 0),
            "timezone":  r.get("timezone", "UTC"),
        }

    # ------------------------------------------------------------------
    # Current / Forecast weather (up to 16 days ahead)
    # ------------------------------------------------------------------
    def fetch_forecast(
        self,
        lat: float,
        lon: float,
        timezone: str = "auto",
        forecast_days: int = 1,
    ) -> Dict[str, Any]:
        """
        Fetch hourly forecast weather data.

        Returns a list of hourly WeatherSnapshot dicts.
        """
        params = {
            "latitude":      lat,
            "longitude":     lon,
            "hourly":        ",".join(_HOURLY_VARS),
            "timezone":      timezone,
            "forecast_days": forecast_days,
        }
        resp = self.session.get(_FORECAST_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()
        logger.info(
            "Forecast fetched for (%.4f, %.4f) — %d hours.",
            lat, lon, len(raw.get("hourly", {}).get("time", [])),
        )
        return self._parse_hourly(raw)

    # ------------------------------------------------------------------
    # Historical weather (up to 3 months ago)
    # ------------------------------------------------------------------
    def fetch_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,   # "YYYY-MM-DD"
        end_date:   str,   # "YYYY-MM-DD"
        timezone: str = "auto",
    ) -> Dict[str, Any]:
        """
        Fetch historical hourly weather data for a date range.
        start_date / end_date format: "YYYY-MM-DD"
        """
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start_date,
            "end_date":   end_date,
            "hourly":     ",".join(_HOURLY_VARS),
            "timezone":   timezone,
        }
        resp = self.session.get(_ARCHIVE_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()
        logger.info(
            "Historical data fetched for (%.4f, %.4f): %s → %s (%d hours).",
            lat, lon, start_date, end_date,
            len(raw.get("hourly", {}).get("time", [])),
        )
        return self._parse_hourly(raw)

    # ------------------------------------------------------------------
    # High-level: fetch by place name (forecast)
    # ------------------------------------------------------------------
    def fetch_weather_for_place(
        self,
        place_name: str,
        forecast_days: int = 1,
    ) -> Dict[str, Any]:
        """One-call: geocode + fetch forecast. Returns full result dict."""
        geo    = self.geocode(place_name)
        result = self.fetch_forecast(
            lat=geo["lat"],
            lon=geo["lon"],
            timezone=geo.get("timezone", "auto"),
            forecast_days=forecast_days,
        )
        result["location"] = geo
        return result

    # ------------------------------------------------------------------
    # Get a single hour snapshot (closest to a given datetime)
    # ------------------------------------------------------------------
    def get_snapshot_at(
        self,
        place_name: str,
        dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Return a single WeatherSnapshot dict for a specific datetime.
        Defaults to the current hour if dt is not provided.
        """
        dt = dt or datetime.utcnow()
        result = self.fetch_weather_for_place(place_name, forecast_days=2)
        snapshots = result.get("hourly_snapshots", [])

        # Find the snapshot whose time is closest to dt
        target_str = dt.strftime("%Y-%m-%dT%H:00")
        for snap in snapshots:
            if snap["time"].startswith(target_str[:13]):
                snap["location"] = result["location"]
                return snap

        # Fallback: return first snapshot
        if snapshots:
            snapshots[0]["location"] = result["location"]
            return snapshots[0]

        raise ValueError(f"No weather snapshot found for '{place_name}' at {dt}.")

    # ------------------------------------------------------------------
    # Internal: parse raw Open-Meteo hourly response
    # ------------------------------------------------------------------
    def _parse_hourly(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        hourly = raw.get("hourly", {})
        times  = hourly.get("time", [])
        n      = len(times)

        snapshots: List[Dict[str, Any]] = []
        for i in range(n):
            wcode = int(hourly.get("weather_code", [0] * n)[i] or 0)
            precip = float(hourly.get("precipitation",   [0] * n)[i] or 0)
            rain   = float(hourly.get("rain",            [0] * n)[i] or 0)

            snap = {
                # ── Time ──────────────────────────────────────────────
                "time":                   times[i],

                # ── Temperature ───────────────────────────────────────
                "temperature_c":          _safe_float(hourly, "temperature_2m",      i),

                # ── Precipitation ─────────────────────────────────────
                "rainfall_mm_1h":         rain,
                "rainfall_mm_24h":        None,   # filled in post-process below
                "snowfall_cm":            _safe_float(hourly, "snowfall",             i),
                "snow_depth_m":           _safe_float(hourly, "snow_depth",           i),
                "total_precipitation_mm": precip,

                # ── Humidity / Pressure ───────────────────────────────
                "relative_humidity_pct":  _safe_float(hourly, "relative_humidity_2m", i),
                "surface_pressure_hpa":   _safe_float(hourly, "surface_pressure",     i),

                # ── Wind ──────────────────────────────────────────────
                "wind_speed_kmh":         _safe_float(hourly, "wind_speed_10m",       i),
                "wind_direction_deg":     _safe_float(hourly, "wind_direction_10m",   i),

                # ── Visibility / Cloud ────────────────────────────────
                "visibility_m":           _safe_float(hourly, "visibility",           i),
                "cloud_cover_pct":        _safe_float(hourly, "cloud_cover",          i),

                # ── Condition ─────────────────────────────────────────
                "weather_code":           wcode,
                "weather_condition":      WMO_CODES.get(wcode, "unknown"),
                "is_day":                 bool(hourly.get("is_day", [1] * n)[i]),

                # ── SUMO friction hint (will be refined by SA-2 model) -
                "road_condition":         _wmo_to_road_condition(wcode),
            }
            snapshots.append(snap)

        # Back-fill rainfall_mm_24h as rolling 24h sum
        rain_vals = [s["rainfall_mm_1h"] for s in snapshots]
        for i, snap in enumerate(snapshots):
            start = max(0, i - 23)
            snap["rainfall_mm_24h"] = round(sum(rain_vals[start: i + 1]), 2)

        return {
            "hourly_snapshots": snapshots,
            "units": {
                "temperature_c":         "°C",
                "rainfall_mm_1h":        "mm",
                "rainfall_mm_24h":       "mm (rolling 24h)",
                "snowfall_cm":           "cm",
                "snow_depth_m":          "m",
                "wind_speed_kmh":        "km/h",
                "wind_direction_deg":    "°",
                "visibility_m":          "m",
                "relative_humidity_pct": "%",
                "surface_pressure_hpa":  "hPa",
                "cloud_cover_pct":       "%",
            },
            "meta": {
                "latitude":  raw.get("latitude"),
                "longitude": raw.get("longitude"),
                "timezone":  raw.get("timezone"),
                "elevation": raw.get("elevation"),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(hourly: dict, key: str, i: int) -> Optional[float]:
    vals = hourly.get(key)
    if vals is None or i >= len(vals) or vals[i] is None:
        return None
    return float(vals[i])


def _wmo_to_road_condition(code: int) -> str:
    """Map WMO weather code to a simplified road surface condition."""
    if code == 0:                   return "dry"
    if code in (1, 2, 3):          return "dry"
    if code in (45, 48):           return "wet"       # fog / rime
    if code in (51, 53, 55):       return "wet"       # drizzle
    if code in (61, 63):           return "wet"       # rain
    if code == 65:                  return "flooded"   # heavy rain
    if code in (71, 73, 75, 77):   return "snowy"
    if code in (80, 81):           return "wet"
    if code == 82:                  return "flooded"
    if code in (85, 86):           return "snowy"
    if code in (95, 96, 99):       return "wet"
    return "unknown"
