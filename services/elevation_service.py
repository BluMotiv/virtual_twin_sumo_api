"""
services/elevation_service.py
=============================
Async elevation (altitude) lookup.

Sources (tried in order):
  1. Open-Meteo built-in elevation (returned with forecast)
  2. Open-Elevation API (open-source, no key)

The elevation is needed for:
  - Air density correction (affects aero drag in SUMO)
  - Pressure adjustment
  - Temperature lapse-rate compensation
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from cachetools import TTLCache

from utils.config import get_settings

logger = logging.getLogger(__name__)


class ElevationService:
    """Fetch terrain elevation for a lat/lon coordinate."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._url = cfg.open_elevation_url
        self._timeout = cfg.open_meteo_timeout
        self._cache: TTLCache = TTLCache(maxsize=512, ttl=86400)  # 24h cache

    async def get_elevation(self, lat: float, lon: float) -> float:
        """
        Return elevation in metres above sea level.

        Falls back to 0.0 if the service is unavailable.
        """
        cache_key = f"elev:{lat:.4f},{lon:.4f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            elevation = await self._from_open_elevation(lat, lon)
        except Exception as exc:
            logger.warning("Elevation lookup failed, defaulting to 0m: %s", exc)
            elevation = 0.0

        self._cache[cache_key] = elevation
        return elevation

    async def _from_open_elevation(self, lat: float, lon: float) -> float:
        """Query the Open-Elevation API."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                self._url,
                params={"locations": f"{lat},{lon}"},
            )
            resp.raise_for_status()

        results = resp.json().get("results", [])
        if results:
            elev = results[0].get("elevation", 0.0)
            logger.info("Elevation (%.4f, %.4f) = %.1f m", lat, lon, elev)
            return float(elev)
        return 0.0
