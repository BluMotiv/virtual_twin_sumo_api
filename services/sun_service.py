"""
services/sun_service.py
========================
Solar radiation / sun-load estimation.

Open-Meteo provides `shortwave_radiation` and `direct_normal_irradiance`
in its hourly forecast.  This service wraps that data and adds a SUMO-
friendly "sun_load_factor" (0.0–1.0) used by the rendering and
thermal models.

If the forecast snapshot lacks radiation fields we fall back to a
simple astronomical model based on lat/lon/datetime.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SunService:
    """Compute solar load parameters from weather snapshot or astronomically."""

    def compute_sun_load(
        self,
        snapshot: Optional[Dict[str, Any]],
        lat: float,
        lon: float,
        dt: datetime,
    ) -> Dict[str, Any]:
        """
        Return a dict with:
            shortwave_radiation_wm2   — surface shortwave radiation (W/m²)
            direct_normal_irradiance  — DNI (W/m²)
            sun_load_factor           — normalised 0.0–1.0 for SUMO
            solar_elevation_deg       — sun elevation angle
            is_day                    — boolean
        """
        # Try to use real data from the weather snapshot first
        sw = None
        dni = None
        is_day = True

        if snapshot:
            sw = snapshot.get("shortwave_radiation")
            dni = snapshot.get("direct_normal_irradiance")
            is_day = snapshot.get("is_day", True)

        # Fall back to astronomical estimate
        solar_elev = self._solar_elevation(lat, lon, dt)
        if sw is None:
            sw = self._estimate_radiation(solar_elev, lat)
        if dni is None:
            dni = sw * 0.75 if sw > 0 else 0.0

        is_day = solar_elev > 0
        sun_load_factor = min(1.0, max(0.0, sw / 1000.0))  # normalise to [0, 1]

        return {
            "shortwave_radiation_wm2": round(sw, 2),
            "direct_normal_irradiance_wm2": round(dni, 2),
            "sun_load_factor": round(sun_load_factor, 4),
            "solar_elevation_deg": round(solar_elev, 2),
            "is_day": is_day,
        }

    # ─────────────────────────────────────────────────────────────
    # Simple solar elevation model
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _solar_elevation(lat: float, lon: float, dt: datetime) -> float:
        """
        Approximate solar elevation angle in degrees.
        Based on the simplified astronomical model.
        """
        doy = dt.timetuple().tm_yday
        # Solar declination (Spencer, 1971)
        B = math.radians((360 / 365.0) * (doy - 81))
        declination = math.radians(
            23.45 * math.sin(B)
        )
        # Hour angle (degrees, 15° per hour from solar noon)
        solar_noon_offset = lon / 15.0  # rough timezone-free offset
        hour_angle = math.radians(15.0 * (dt.hour + dt.minute / 60.0 - 12.0 + solar_noon_offset))

        lat_rad = math.radians(lat)
        sin_elev = (
            math.sin(lat_rad) * math.sin(declination)
            + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
        )
        return math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    @staticmethod
    def _estimate_radiation(solar_elevation_deg: float, lat: float) -> float:
        """Estimate surface shortwave radiation from solar elevation."""
        if solar_elevation_deg <= 0:
            return 0.0
        # Clear-sky model: ~1000 W/m² at zenith, reduced by air mass
        air_mass = 1.0 / max(math.sin(math.radians(solar_elevation_deg)), 0.05)
        # Simple atmospheric extinction
        radiation = 1361.0 * 0.7 ** (air_mass ** 0.678)
        return max(0.0, radiation * math.sin(math.radians(solar_elevation_deg)))
