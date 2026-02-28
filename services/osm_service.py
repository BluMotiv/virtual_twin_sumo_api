"""
services/osm_service.py
========================
OpenStreetMap road network builder for SUMO scenarios.

Responsibilities
----------------
1. compute_bbox()       — 1 km x 1 km bounding box around a lat/lon point
2. download_osm_data()  — Download raw OSM XML from Overpass API
3. run_netconvert()     — Convert OSM XML to SUMO .net.xml  (subprocess)
4. fetch_road_context() — Quick point-query for nearest road metadata
5. build_network()      — Master coroutine that chains 1->2->3

Altitude is passed to netconvert as a z-offset so every node in the
generated network carries the correct terrain elevation.
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from cachetools import TTLCache

from utils.config import get_settings

logger = logging.getLogger(__name__)

_OVERPASS_URL    = "https://overpass-api.de/api/interpreter"
_OVERPASS_TIMEOUT = 60
_CONTEXT_TIMEOUT  = 20

# 500 m in latitude degrees (constant everywhere)
_LAT_DEG_500M = 500.0 / 111_320.0   # approx 0.004493 degrees

def _lon_deg_500m(lat: float) -> float:
    """Degrees of longitude equivalent to 500 m at lat."""
    cos_lat = math.cos(math.radians(lat))
    return (500.0 / (111_320.0 * cos_lat)) if cos_lat > 1e-6 else _LAT_DEG_500M

_cache: TTLCache = TTLCache(maxsize=256, ttl=get_settings().cache_ttl)


def compute_bbox(lat: float, lon: float) -> Dict[str, float]:
    """
    Return a 1 km x 1 km bounding box centred on (lat, lon).
    Keys: south, west, north, east (decimal degrees).
    """
    dlat = _LAT_DEG_500M
    dlon = _lon_deg_500m(lat)
    return {
        "south": round(lat - dlat, 6),
        "west":  round(lon - dlon, 6),
        "north": round(lat + dlat, 6),
        "east":  round(lon + dlon, 6),
    }


async def download_osm_data(bbox: Dict[str, float], osm_path: Path) -> bool:
    """
    Download OSM road network for bbox from Overpass and save to osm_path.
    Returns True on success.
    """
    south, west, north, east = (
        bbox["south"], bbox["west"], bbox["north"], bbox["east"]
    )
    query = (
        f"[out:xml][timeout:{_OVERPASS_TIMEOUT}];\n"
        f"(\n"
        f'  way["highway"]({south},{west},{north},{east});\n'
        f"  node(w);\n"
        f'  relation["highway"]({south},{west},{north},{east});\n'
        f");\n"
        f"out body;\n>;\nout skel qt;\n"
    )
    logger.info("Downloading OSM bbox S=%.6f W=%.6f N=%.6f E=%.6f", south, west, north, east)
    try:
        async with httpx.AsyncClient(timeout=_OVERPASS_TIMEOUT + 10) as client:
            resp = await client.post(
                _OVERPASS_URL,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
        content = resp.content
        if len(content) < 500:
            logger.warning("OSM response too small (%d bytes)", len(content))
            return False
        osm_path.write_bytes(content)
        logger.info("OSM saved -> %s (%.1f KB)", osm_path, len(content) / 1024)
        return True
    except Exception as exc:
        logger.error("OSM download failed: %s", exc)
        return False


def run_netconvert(osm_path: Path, net_path: Path, elevation_m: float = 0.0) -> bool:
    """
    Convert OSM file to SUMO .net.xml using netconvert.
    Embeds elevation_m as a z-offset on every network node.
    Returns True on success.
    """
    cmd = [
        "netconvert",
        "--osm-files",            str(osm_path),
        "--output-file",          str(net_path),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--keep-edges.by-vclass", "passenger,bus,truck,bicycle,pedestrian",
        "--osm.sidewalks",        "false",
        "--osm.crossings",        "false",
    ]
    if elevation_m:
        cmd += ["--offset.z", str(round(elevation_m, 2))]

    logger.info("Running netconvert (elevation_m=%.1f)...", elevation_m)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("netconvert failed (rc=%d):\n%s", result.returncode, result.stderr[-2000:])
            return False
        if not net_path.exists() or net_path.stat().st_size < 1000:
            logger.error("netconvert produced empty/missing file: %s", net_path)
            return False
        logger.info("Network built -> %s (%.1f KB)", net_path, net_path.stat().st_size / 1024)
        return True
    except subprocess.TimeoutExpired:
        logger.error("netconvert timed out after 120 s")
        return False
    except FileNotFoundError:
        logger.error("netconvert not found - install SUMO and ensure it is on PATH")
        return False


async def fetch_road_context(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch nearest road metadata for a point via Overpass (TTL-cached).
    Returns road_type, road_surface, speed_limit, lanes, road_name,
    road_types list, has_traffic_lights, speed_limits list, road_count.
    """
    cache_key = f"road_ctx:{lat:.4f},{lon:.4f}"
    if cache_key in _cache:
        return _cache[cache_key]

    defaults: Dict[str, Any] = {
        "road_types": ["unclassified"], "has_traffic_lights": False,
        "speed_limits": [], "road_count": 0,
        "road_type": "unclassified", "road_surface": "paved",
        "speed_limit": 50, "lanes": 2, "road_name": "Unknown",
    }

    query = (
        f"[out:json][timeout:15];\n"
        f"(\n"
        f'  way["highway"](around:150,{lat},{lon});\n'
        f'  node["highway"="traffic_signals"](around:300,{lat},{lon});\n'
        f");\nout tags;\n"
    )

    try:
        async with httpx.AsyncClient(timeout=_CONTEXT_TIMEOUT) as client:
            resp = await client.post(
                _OVERPASS_URL,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()

        elements = data.get("elements", [])
        road_types: set = set()
        speed_limits: list = []
        has_signals = False
        road_count = 0
        first_road_tags: Dict = {}

        for el in elements:
            tags = el.get("tags", {})
            if el.get("type") == "node" and tags.get("highway") == "traffic_signals":
                has_signals = True
                continue
            hw = tags.get("highway")
            if hw:
                road_types.add(hw)
                road_count += 1
                if not first_road_tags:
                    first_road_tags = tags
            ms = tags.get("maxspeed", "")
            if ms:
                speed_limits.append(ms)

        speed_int = 50
        if speed_limits:
            try:
                speed_int = int("".join(filter(str.isdigit, speed_limits[0])))
            except (ValueError, TypeError):
                pass

        result: Dict[str, Any] = {
            "road_types":         sorted(road_types),
            "has_traffic_lights": has_signals,
            "speed_limits":       speed_limits,
            "road_count":         road_count,
            "road_type":          first_road_tags.get("highway",  "unclassified"),
            "road_surface":       first_road_tags.get("surface",  "paved"),
            "speed_limit":        speed_int,
            "lanes":              int(first_road_tags.get("lanes", 2)),
            "road_name":          first_road_tags.get("name",
                                   first_road_tags.get("ref", "Unknown")),
        }
        logger.info(
            "Road context (%.4f, %.4f): %d roads, signals=%s, surface=%s",
            lat, lon, road_count, has_signals, result["road_surface"],
        )
        _cache[cache_key] = result
        return result

    except Exception as exc:
        logger.warning("Road context fetch failed: %s - using defaults", exc)
        return defaults


async def build_network(
    lat: float,
    lon: float,
    output_dir: Path,
    elevation_m: float = 0.0,
) -> Optional[Path]:
    """
    Build a 1 km x 1 km SUMO road network centred on (lat, lon).

    Steps
    -----
    1. compute_bbox()       - bounding box
    2. download_osm_data()  - Overpass download
    3. run_netconvert()     - OSM -> .net.xml with elevation z-offset

    Returns Path to .net.xml, or None on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox     = compute_bbox(lat, lon)
    osm_path = output_dir / "network.osm"
    net_path = output_dir / "network.net.xml"

    if not await download_osm_data(bbox, osm_path):
        logger.error("OSM download failed - network will not be generated")
        return None

    if not run_netconvert(osm_path, net_path, elevation_m=elevation_m):
        logger.error("netconvert failed - network will not be generated")
        return None

    return net_path


# Legacy class - kept for backward compatibility
class OSMService:
    """Thin wrapper kept for backward compatibility with existing main.py."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._timeout = _CONTEXT_TIMEOUT

    async def get_road_context(
        self, lat: float, lon: float, radius_m: int = 1000,
    ) -> Dict[str, Any]:
        return await fetch_road_context(lat, lon)
