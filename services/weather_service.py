"""
services/weather_service.py
============================
Async weather data service using Open-Meteo.

Supports:
  - Geocoding (place name → lat/lon)
  - Hourly forecast (up to 16 days)
  - Historical archive
  - Snapshot at specific datetime
  - Historical lag extraction for ML feature engineering

No API key required.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from cachetools import TTLCache

from utils.config import get_settings

logger = logging.getLogger(__name__)

# ── WMO weather code lookup ─────────────────────────────────────
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

# ── Hourly variables we request from Open-Meteo ──
_HOURLY_VARS = ",".join([
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
    "shortwave_radiation",          # solar irradiance (W/m²)
    "direct_normal_irradiance",     # for SUMO sun-load
])


class WeatherService:
    """Async Open-Meteo weather data client with TTL caching."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._forecast_url = cfg.open_meteo_forecast_url
        self._archive_url = cfg.open_meteo_archive_url
        self._geocode_url = cfg.open_meteo_geocode_url
        self._timeout = cfg.open_meteo_timeout
        self._cache: TTLCache = TTLCache(maxsize=256, ttl=cfg.cache_ttl)

    # ─────────────────────────────────────────────────────────────
    # Geocoding
    # ─────────────────────────────────────────────────────────────
    async def geocode(self, place_name: str) -> Dict[str, Any]:
        """Resolve a place name to lat/lon/elevation via Open-Meteo."""
        cache_key = f"geo:{place_name.lower().strip()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._geocode_url,
                params={"name": place_name, "count": 1, "language": "en", "format": "json"},
            )
            resp.raise_for_status()

        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"Geocoding failed: no results for '{place_name}'.")

        r = results[0]
        geo = {
            "name": r.get("name", place_name),
            "country": r.get("country", ""),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "elevation": r.get("elevation", 0),
            "timezone": r.get("timezone", "UTC"),
        }
        self._cache[cache_key] = geo
        logger.info("Geocoded '%s' → lat=%.4f, lon=%.4f", place_name, geo["lat"], geo["lon"])
        return geo

    # ─────────────────────────────────────────────────────────────
    # Forecast (up to 16 days)
    # ─────────────────────────────────────────────────────────────
    async def fetch_forecast(
        self,
        lat: float,
        lon: float,
        timezone: str = "auto",
        forecast_days: int = 7,
    ) -> Dict[str, Any]:
        """Fetch hourly forecast data and return parsed snapshots."""
        cache_key = f"fc:{lat:.4f},{lon:.4f},{forecast_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": _HOURLY_VARS,
            "timezone": timezone,
            "forecast_days": min(forecast_days, 16),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._forecast_url, params=params)
            resp.raise_for_status()

        result = self._parse_hourly(resp.json())
        self._cache[cache_key] = result
        logger.info(
            "Forecast fetched: (%.4f, %.4f) — %d snapshots",
            lat, lon, len(result.get("hourly_snapshots", [])),
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Historical archive
    # ─────────────────────────────────────────────────────────────
    async def fetch_historical(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        timezone: str = "auto",
    ) -> Dict[str, Any]:
        """Fetch historical hourly weather for a date range (YYYY-MM-DD)."""
        cache_key = f"hist:{lat:.4f},{lon:.4f},{start_date},{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": _HOURLY_VARS,
            "timezone": timezone,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._archive_url, params=params)
            resp.raise_for_status()

        result = self._parse_hourly(resp.json())
        self._cache[cache_key] = result
        logger.info(
            "Historical fetched: (%.4f, %.4f) %s→%s — %d snapshots",
            lat, lon, start_date, end_date,
            len(result.get("hourly_snapshots", [])),
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Snapshot at specific datetime
    # ─────────────────────────────────────────────────────────────
    async def get_snapshot_at(
        self,
        lat: float,
        lon: float,
        dt: datetime,
        timezone: str = "auto",
    ) -> Optional[Dict[str, Any]]:
        """
        Return the single hourly snapshot closest to *dt*.

        Uses forecast if dt is within the forecast window,
        otherwise falls back to historical archive.
        """
        today = date.today()
        target_date = dt.date()
        delta = (target_date - today).days

        if delta >= 0 and delta <= 16:
            # Forecast mode
            data = await self.fetch_forecast(
                lat, lon, timezone=timezone, forecast_days=max(delta + 1, 1),
            )
        else:
            # Historical mode
            ds = target_date.isoformat()
            data = await self.fetch_historical(lat, lon, ds, ds, timezone)

        target_hour = dt.strftime("%Y-%m-%dT%H")
        for snap in data.get("hourly_snapshots", []):
            if snap["time"].startswith(target_hour):
                return snap

        # Fallback: return first snapshot
        snaps = data.get("hourly_snapshots", [])
        return snaps[0] if snaps else None

    # ─────────────────────────────────────────────────────────────
    # Historical lag data for ML feature engineering
    # ─────────────────────────────────────────────────────────────
    async def get_historical_lags(
        self,
        lat: float,
        lon: float,
        target_date: date,
        lookback_days: int = 30,
        timezone: str = "auto",
    ) -> Dict[str, List[float]]:
        """
        Fetch historical data for *lookback_days* before target_date
        and return daily aggregated lag values for ML features.

        **Key logic**: The Open-Meteo Archive only serves data up to
        yesterday.  When target_date is in the future we clamp end_date
        to yesterday so we always pull the freshest real observations.
        The ML feature builder gracefully handles shorter-than-ideal
        lag windows (falls back to climatological estimates).

        Returns dict with keys:
            temps       — daily mean temperatures (most recent first)
            precip      — daily total precipitation (most recent first)
            wind        — hourly wind speeds (most recent first)
            sunload     — hourly solar radiation (most recent first)
            wind_dir    — last known wind direction
        """
        from datetime import date as _date

        yesterday = _date.today() - timedelta(days=1)

        # end_date: ideally the day before the target, but never past yesterday
        ideal_end = target_date - timedelta(days=1)
        end = min(ideal_end, yesterday)

        # start_date: lookback_days before end (not before target)
        start = end - timedelta(days=lookback_days)

        # Sanity — if end < start the range is empty
        if end < start:
            logger.warning(
                "Historical lag range is empty (end=%s < start=%s) — returning defaults",
                end, start,
            )
            return {"temps": [], "precip": [], "wind": [], "sunload": [], "wind_dir": 180.0}

        days_available = (end - start).days + 1
        logger.info(
            "Fetching %d days of historical lags:  %s → %s  (target=%s, yesterday=%s)",
            days_available, start, end, target_date, yesterday,
        )

        try:
            data = await self.fetch_historical(
                lat, lon,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                timezone=timezone,
            )
        except Exception as exc:
            logger.warning("Failed to fetch historical lags: %s", exc)
            return {"temps": [], "precip": [], "wind": [], "sunload": [], "wind_dir": 180.0}

        snaps = data.get("hourly_snapshots", [])
        if not snaps:
            return {"temps": [], "precip": [], "wind": [], "sunload": [], "wind_dir": 180.0}

        # ── Aggregate daily temps & precip ──
        from collections import defaultdict
        daily_temps: Dict[str, List[float]] = defaultdict(list)
        daily_precip: Dict[str, float] = defaultdict(float)

        hourly_wind: List[float] = []
        hourly_sunload: List[float] = []
        last_wind_dir = 180.0

        for s in snaps:
            day_key = s["time"][:10]
            if s.get("temperature_c") is not None:
                daily_temps[day_key].append(s["temperature_c"])
            daily_precip[day_key] += s.get("rainfall_mm_1h", 0.0) or 0.0
            if s.get("wind_speed_kmh") is not None:
                hourly_wind.append(s["wind_speed_kmh"])
            if s.get("shortwave_radiation") is not None:
                hourly_sunload.append(s["shortwave_radiation"])
            elif s.get("cloud_cover_pct") is not None:
                # Estimate solar from cloud cover (crude)
                hourly_sunload.append(max(0.0, 800.0 * (1 - s["cloud_cover_pct"] / 100)))
            if s.get("wind_direction_deg") is not None:
                last_wind_dir = s["wind_direction_deg"]

        # Sort by date descending (most recent first)
        sorted_days = sorted(daily_temps.keys(), reverse=True)
        temps = [float(sum(daily_temps[d]) / len(daily_temps[d])) for d in sorted_days]
        precip = [daily_precip[d] for d in sorted_days]

        # Hourly lists — reverse so most recent is first
        hourly_wind.reverse()
        hourly_sunload.reverse()

        return {
            "temps": temps,
            "precip": precip,
            "wind": hourly_wind,
            "sunload": hourly_sunload,
            "wind_dir": last_wind_dir,
        }

    # ─────────────────────────────────────────────────────────────
    # Internal: parse Open-Meteo hourly JSON
    # ─────────────────────────────────────────────────────────────
    def _parse_hourly(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw Open-Meteo JSON to a list of snapshot dicts."""
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        n = len(times)

        snapshots: List[Dict[str, Any]] = []
        for i in range(n):
            wcode = int(self._safe(hourly, "weather_code", i) or 0)
            rain = self._safe(hourly, "rain", i) or 0.0

            snap = {
                "time": times[i],
                "temperature_c": self._safe(hourly, "temperature_2m", i),
                "rainfall_mm_1h": rain,
                "rainfall_mm_24h": None,
                "snowfall_cm": self._safe(hourly, "snowfall", i),
                "snow_depth_m": self._safe(hourly, "snow_depth", i),
                "total_precipitation_mm": self._safe(hourly, "precipitation", i) or 0.0,
                "relative_humidity_pct": self._safe(hourly, "relative_humidity_2m", i),
                "surface_pressure_hpa": self._safe(hourly, "surface_pressure", i),
                "wind_speed_kmh": self._safe(hourly, "wind_speed_10m", i),
                "wind_direction_deg": self._safe(hourly, "wind_direction_10m", i),
                "visibility_m": self._safe(hourly, "visibility", i),
                "cloud_cover_pct": self._safe(hourly, "cloud_cover", i),
                "shortwave_radiation": self._safe(hourly, "shortwave_radiation", i),
                "direct_normal_irradiance": self._safe(hourly, "direct_normal_irradiance", i),
                "weather_code": wcode,
                "weather_condition": WMO_CODES.get(wcode, "unknown"),
                "is_day": bool(self._safe(hourly, "is_day", i) or 0),
                "road_condition": self._wmo_to_road(wcode),
            }
            snapshots.append(snap)

        # Back-fill 24h rolling rain
        rain_vals = [s["rainfall_mm_1h"] for s in snapshots]
        for i, snap in enumerate(snapshots):
            snap["rainfall_mm_24h"] = round(sum(rain_vals[max(0, i - 23): i + 1]), 2)

        return {
            "hourly_snapshots": snapshots,
            "meta": {
                "latitude": raw.get("latitude"),
                "longitude": raw.get("longitude"),
                "timezone": raw.get("timezone"),
                "elevation": raw.get("elevation"),
            },
        }

    @staticmethod
    def _safe(hourly: dict, key: str, i: int) -> Optional[float]:
        vals = hourly.get(key)
        if vals is None or i >= len(vals) or vals[i] is None:
            return None
        return float(vals[i])

    @staticmethod
    def _wmo_to_road(code: int) -> str:
        if code == 0:                   return "dry"
        if code in (1, 2, 3):          return "dry"
        if code in (45, 48):           return "wet"
        if code in (51, 53, 55):       return "wet"
        if code in (61, 63):           return "wet"
        if code == 65:                  return "flooded"
        if code in (71, 73, 75, 77):   return "snowy"
        if code in (80, 81):           return "wet"
        if code == 82:                  return "flooded"
        if code in (85, 86):           return "snowy"
        if code in (95, 96, 99):       return "wet"
        return "unknown"
