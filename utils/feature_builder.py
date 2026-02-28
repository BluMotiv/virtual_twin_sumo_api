"""
utils/feature_builder.py
========================
Constructs ML-ready feature vectors for each model from raw
environmental data + request metadata.

Each builder function returns a dict keyed by the exact feature
names the corresponding model expects.

Feature Specs (from pickle inspection):
──────────────────────────────────────────────────────────────
Temperature  (7 features):
  DOY_SIN, DOY_COS, T2M_LAG_1, T2M_LAG_7, T2M_LAG_30, LAT, LON

Precipitation (16 features):
  YEAR, MO, DY, T2M, WS2M, ALLSKY_SFC_SW_DWN,
  lag_1, lag_3, lag_7, lag_14, roll_sum_7, roll_sum_14,
  month, dayofyear, latitude, longitude

Sunload (11 features):
  latitude, longitude, hour_sin, hour_cos, month_sin, month_cos,
  dayofweek, dayofyear, sunload_lag_1h, sunload_mean_3h, sunload_max_24h

Wind Speed (13 features):
  latitude, longitude, hour_sin, hour_cos, month_sin, month_cos,
  dayofweek, dayofyear, wind_dir_sin, wind_dir_cos,
  wind_lag_1h, wind_mean_3h, wind_max_24h
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────
# Cyclical encoding helpers
# ──────────────────────────────────────────────────────────────────
def _sin_cos(value: float, period: float) -> tuple[float, float]:
    """Return (sin, cos) encoding for a cyclical feature."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


# ──────────────────────────────────────────────────────────────────
# Temperature feature vector
# ──────────────────────────────────────────────────────────────────
def build_temperature_features(
    dt: datetime,
    lat: float,
    lon: float,
    historical_temps: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Build the 7-feature vector for the temperature XGBoost model.

    Parameters
    ----------
    dt : datetime
        Target prediction datetime.
    lat, lon : float
        Location coordinates.
    historical_temps : list[float] | None
        Recent daily mean temperatures (most recent first).
        Index 0 → yesterday, 6 → 7 days ago, 29 → 30 days ago.
        If None, lag features default to seasonal climatological estimates.
    """
    doy = dt.timetuple().tm_yday
    doy_sin, doy_cos = _sin_cos(doy, 365.25)

    # Lag features — use historical if available, else seasonal fallback
    if historical_temps and len(historical_temps) >= 30:
        t2m_lag_1 = historical_temps[0]
        t2m_lag_7 = historical_temps[6]
        t2m_lag_30 = historical_temps[29]
    elif historical_temps and len(historical_temps) >= 7:
        t2m_lag_1 = historical_temps[0]
        t2m_lag_7 = historical_temps[min(6, len(historical_temps) - 1)]
        t2m_lag_30 = historical_temps[-1]  # best available
    elif historical_temps and len(historical_temps) >= 1:
        t2m_lag_1 = historical_temps[0]
        t2m_lag_7 = historical_temps[0]
        t2m_lag_30 = historical_temps[0]
    else:
        # Climatological estimate based on latitude & day-of-year
        t2m_lag_1 = _climatological_temp(lat, doy)
        t2m_lag_7 = t2m_lag_1
        t2m_lag_30 = t2m_lag_1

    return {
        "DOY_SIN": doy_sin,
        "DOY_COS": doy_cos,
        "T2M_LAG_1": t2m_lag_1,
        "T2M_LAG_7": t2m_lag_7,
        "T2M_LAG_30": t2m_lag_30,
        "LAT": lat,
        "LON": lon,
    }


# ──────────────────────────────────────────────────────────────────
# Precipitation feature vector
# ──────────────────────────────────────────────────────────────────
def build_precipitation_features(
    dt: datetime,
    lat: float,
    lon: float,
    temperature: float = 25.0,
    wind_speed: float = 5.0,
    solar_radiation: float = 200.0,
    historical_precip: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Build the 16-feature vector for the precipitation model.

    Parameters
    ----------
    dt : datetime
        Target prediction datetime.
    lat, lon : float
        Location coordinates.
    temperature : float
        Current / predicted temperature (°C).
    wind_speed : float
        Current / predicted wind speed (m/s at 2m).
    solar_radiation : float
        Current / predicted solar radiation (W/m²).
    historical_precip : list[float] | None
        Recent daily precipitation totals (most recent first),
        at least 14 values preferred.
    """
    hp = historical_precip or [0.0] * 14

    # Pad if shorter than 14
    while len(hp) < 14:
        hp.append(0.0)

    return {
        "YEAR": float(dt.year),
        "MO": float(dt.month),
        "DY": float(dt.day),
        "T2M": temperature,
        "WS2M": wind_speed,
        "ALLSKY_SFC_SW_DWN": solar_radiation,
        "lag_1": hp[0],
        "lag_3": hp[2] if len(hp) > 2 else 0.0,
        "lag_7": hp[6] if len(hp) > 6 else 0.0,
        "lag_14": hp[13] if len(hp) > 13 else 0.0,
        "roll_sum_7": sum(hp[:7]),
        "roll_sum_14": sum(hp[:14]),
        "month": float(dt.month),
        "dayofyear": float(dt.timetuple().tm_yday),
        "latitude": lat,
        "longitude": lon,
    }


# ──────────────────────────────────────────────────────────────────
# Sunload feature vector
# ──────────────────────────────────────────────────────────────────
def build_sunload_features(
    dt: datetime,
    lat: float,
    lon: float,
    historical_sunload: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Build the 11-feature vector for the sunload XGBoost model.

    Parameters
    ----------
    dt : datetime
        Target prediction datetime.
    lat, lon : float
        Location coordinates.
    historical_sunload : list[float] | None
        Recent hourly sunload values (most recent first).
        Index 0 → last hour, etc.  At least 24 values preferred.
    """
    hour_sin, hour_cos = _sin_cos(dt.hour, 24)
    month_sin, month_cos = _sin_cos(dt.month, 12)
    doy = dt.timetuple().tm_yday

    hs = historical_sunload or []
    lag_1h = hs[0] if len(hs) >= 1 else _default_sunload(lat, dt)
    mean_3h = float(np.mean(hs[:3])) if len(hs) >= 3 else lag_1h
    max_24h = float(np.max(hs[:24])) if len(hs) >= 24 else lag_1h * 1.3

    return {
        "latitude": lat,
        "longitude": lon,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dayofweek": float(dt.weekday()),
        "dayofyear": float(doy),
        "sunload_lag_1h": lag_1h,
        "sunload_mean_3h": mean_3h,
        "sunload_max_24h": max_24h,
    }


# ──────────────────────────────────────────────────────────────────
# Wind Speed feature vector
# ──────────────────────────────────────────────────────────────────
def build_wind_speed_features(
    dt: datetime,
    lat: float,
    lon: float,
    wind_direction_deg: float = 180.0,
    historical_wind: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Build the 13-feature vector for the wind speed XGBoost model.

    Parameters
    ----------
    dt : datetime
        Target prediction datetime.
    lat, lon : float
        Location coordinates.
    wind_direction_deg : float
        Wind direction in degrees (0-360).
    historical_wind : list[float] | None
        Recent hourly wind speed values (most recent first).
        At least 24 values preferred.
    """
    hour_sin, hour_cos = _sin_cos(dt.hour, 24)
    month_sin, month_cos = _sin_cos(dt.month, 12)
    doy = dt.timetuple().tm_yday

    dir_rad = math.radians(wind_direction_deg)
    wind_dir_sin = math.sin(dir_rad)
    wind_dir_cos = math.cos(dir_rad)

    hw = historical_wind or []
    lag_1h = hw[0] if len(hw) >= 1 else 5.0  # default 5 km/h
    mean_3h = float(np.mean(hw[:3])) if len(hw) >= 3 else lag_1h
    max_24h = float(np.max(hw[:24])) if len(hw) >= 24 else lag_1h * 1.5

    return {
        "latitude": lat,
        "longitude": lon,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dayofweek": float(dt.weekday()),
        "dayofyear": float(doy),
        "wind_dir_sin": wind_dir_sin,
        "wind_dir_cos": wind_dir_cos,
        "wind_lag_1h": lag_1h,
        "wind_mean_3h": mean_3h,
        "wind_max_24h": max_24h,
    }


# ──────────────────────────────────────────────────────────────────
# Private climatological fallbacks
# ──────────────────────────────────────────────────────────────────
def _climatological_temp(lat: float, doy: int) -> float:
    """
    Very rough seasonal temperature estimate (°C).
    Uses latitude and day-of-year with a simple sinusoidal model.
    """
    # Base temperature decreases with |latitude|
    base = 30.0 - 0.5 * abs(lat)
    # Northern hemisphere: warmest ~doy 200; Southern: ~doy 15
    if lat >= 0:
        phase = (doy - 200) / 365.25
    else:
        phase = (doy - 15) / 365.25
    amplitude = 15.0 * (abs(lat) / 60.0)  # larger swing at higher latitudes
    return base + amplitude * math.cos(2 * math.pi * phase)


def _default_sunload(lat: float, dt: datetime) -> float:
    """Rough solar irradiance estimate (W/m²) based on hour and latitude."""
    if dt.hour < 6 or dt.hour > 18:
        return 0.0
    # Peak at solar noon (~12:00)
    hour_factor = math.cos(math.pi * (dt.hour - 12) / 12)
    lat_factor = max(0.2, 1.0 - abs(lat) / 90.0)
    return max(0.0, 800.0 * hour_factor * lat_factor)
