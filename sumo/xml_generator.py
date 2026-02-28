"""
sumo/xml_generator.py
=====================
Generate ALL SUMO-compatible files for a scenario.

Output files
------------
1. weather.add.xml        — friction, visibility, wind, precipitation
2. environment.xml        — full env spec incl. elevation + bbox
3. vehicle_types.add.xml  — speed/behaviour adjustments (elevation-aware)
4. traffic_lights.add.xml — clearance time adjustments for bad weather
5. routes.rou.xml         — random vehicle routes (via randomTrips.py)
6. scenario.sumocfg       — master SUMO config tying all files together
7. scenario_config.xml    — BluFleet metadata manifest

Altitude usage
--------------
  - environment.xml        <location elevation_m="...">
  - vehicle_types.add.xml  engine decel factor for high-altitude
  - scenario.sumocfg       <location altitude_m="...">
  - scenario_config.xml    <info elevation_m="...">
  - network.net.xml        z-offset applied by netconvert (osm_service)

References
----------
  https://sumo.dlr.de/docs/Simulation/Weather.html
  https://sumo.dlr.de/docs/sumo.html#format_of_additional_files
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.dom.minidom import parseString

from lxml import etree

from utils.config import get_settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Road friction model
# ──────────────────────────────────────────────────────────────────
def _compute_friction(
    temperature_c: Optional[float],
    rain_mm: Optional[float],
    snow_cm: Optional[float],
    road_condition: str,
) -> float:
    """
    Compute road surface friction coefficient (μ) for SUMO.

    Dry asphalt   : ~0.80
    Wet           : ~0.50-0.65
    Snowy / icy   : ~0.20-0.35
    Flooded       : ~0.30-0.40
    """
    base = 0.80  # dry asphalt

    if road_condition == "snowy":
        base = 0.25
        if temperature_c is not None and temperature_c < -5:
            base = 0.15  # black ice risk
    elif road_condition == "flooded":
        base = 0.35
    elif road_condition == "wet":
        base = 0.55
        if rain_mm and rain_mm > 10:
            base = 0.45  # heavy rain
    elif road_condition == "dry":
        base = 0.80

    # Temperature correction: very cold → ice
    if temperature_c is not None and temperature_c < 0 and road_condition != "dry":
        base = min(base, 0.20)

    return round(max(0.10, min(1.0, base)), 3)


# ──────────────────────────────────────────────────────────────────
# Visibility → SUMO driver imperfection factor
# ──────────────────────────────────────────────────────────────────
def _visibility_to_imperfection(visibility_m: Optional[float]) -> float:
    """
    Map visibility (metres) to SUMO driver imperfection [0, 1].
    Lower visibility → higher imperfection → more erratic driving.
    """
    if visibility_m is None or visibility_m >= 10000:
        return 0.3  # normal
    if visibility_m >= 5000:
        return 0.4
    if visibility_m >= 1000:
        return 0.6
    if visibility_m >= 200:
        return 0.8
    return 0.95  # near-zero visibility


# ──────────────────────────────────────────────────────────────────
# Speed factor adjustment for weather
# ──────────────────────────────────────────────────────────────────
def _weather_speed_factor(
    road_condition: str,
    visibility_m: Optional[float],
    wind_speed_kmh: Optional[float],
) -> float:
    """
    Compute a speed reduction factor [0.5, 1.0] due to weather.
    Applied as speedFactor in SUMO vehicle types.
    """
    factor = 1.0

    if road_condition in ("wet", "flooded"):
        factor *= 0.85
    elif road_condition == "snowy":
        factor *= 0.70

    if visibility_m is not None and visibility_m < 1000:
        factor *= max(0.6, visibility_m / 1500)

    if wind_speed_kmh is not None and wind_speed_kmh > 50:
        factor *= 0.90

    return round(max(0.50, factor), 3)


class SUMOXMLGenerator:
    """Generate SUMO-compatible XML files for a weather scenario."""

    def __init__(self) -> None:
        self._output_dir = get_settings().sumo_output_path

    def generate(
        self,
        environment: Dict[str, Any],
        scenario_id: str,
        lat: float,
        lon: float,
        dt: datetime,
        prediction_source: str = "forecast",
        road_context: Optional[Dict[str, Any]] = None,
        net_path: Optional[Path] = None,
    ) -> List[str]:
        """
        Generate all SUMO files for a scenario and return their paths.

        Parameters
        ----------
        environment      : merged env features (weather + ML + elevation + bbox)
        scenario_id      : unique ID string
        lat, lon         : centre coordinates
        dt               : scenario datetime
        prediction_source: "forecast" or "ml_model"
        road_context     : OSM road context dict (for TLS flag)
        net_path         : pre-built .net.xml from osm_service.build_network()

        Returns
        -------
        list[str] — absolute paths of all generated files.
        """
        out_dir = self._output_dir / scenario_id
        out_dir.mkdir(parents=True, exist_ok=True)

        files_generated: List[str] = []

        # 1. weather.add.xml
        weather_path = out_dir / "weather.add.xml"
        self._generate_weather_xml(weather_path, environment, dt)
        files_generated.append(str(weather_path))

        # 2. environment.xml  (elevation + bbox)
        env_path = out_dir / "environment.xml"
        self._generate_environment_xml(env_path, environment, lat, lon, dt, prediction_source)
        files_generated.append(str(env_path))

        # 3. vehicle_types.add.xml  (elevation-aware accel/decel)
        vtype_path = out_dir / "vehicle_types.add.xml"
        self._generate_vehicle_types_xml(vtype_path, environment)
        files_generated.append(str(vtype_path))

        # 4. traffic_lights.add.xml
        tls_path = out_dir / "traffic_lights.add.xml"
        self._generate_tls_xml(tls_path, environment)
        files_generated.append(str(tls_path))

        # 5. routes.rou.xml  (requires network)
        rou_path: Optional[Path] = None
        if net_path and net_path.exists():
            rou_path = self._generate_routes(net_path, out_dir)
            if rou_path:
                files_generated.append(str(rou_path))

        # 6. network.net.xml  (built by osm_service, referenced here)
        if net_path and net_path.exists():
            files_generated.append(str(net_path))

        # 7. scenario.sumocfg  (master config — elevation embedded)
        sumocfg_path = out_dir / "scenario.sumocfg"
        self._generate_sumocfg(
            path=sumocfg_path,
            scenario_id=scenario_id,
            lat=lat, lon=lon, dt=dt,
            env=environment,
            net_path=net_path,
            rou_path=rou_path,
            extra_files=[weather_path, vtype_path],   # tls_path excluded — sidecar only
        )
        files_generated.append(str(sumocfg_path))

        # 8. scenario_config.xml  (BluFleet manifest)
        config_path = out_dir / "scenario_config.xml"
        self._generate_config_xml(
            config_path, scenario_id, lat, lon, dt,
            prediction_source, files_generated, environment,
        )
        files_generated.append(str(config_path))

        logger.info(
            "Generated %d SUMO files in %s", len(files_generated), out_dir
        )
        return files_generated

    # ─────────────────────────────────────────────────────────────
    # 1. Weather additional file
    # ─────────────────────────────────────────────────────────────
    def _generate_weather_xml(
        self,
        path: Path,
        env: Dict[str, Any],
        dt: datetime,
    ) -> None:
        """Generate weather.add.xml for SUMO simulation."""
        root = etree.Element("additional")
        root.addprevious(etree.Comment(
            f" Generated by SUMO Scenario API — {datetime.now().isoformat()} "
        ))

        # Weather parameter element
        weather = etree.SubElement(root, "weather")
        weather.set("time", dt.strftime("%Y-%m-%dT%H:%M:%S"))

        # Temperature
        temp = env.get("temperature_c")
        if temp is not None:
            weather.set("temperature", f"{temp:.1f}")

        # Precipitation
        rain = env.get("rainfall_mm_1h", env.get("rain_mm", 0.0))
        weather.set("precipitation", f"{rain:.2f}" if rain else "0.00")
        weather.set("precipitationType", self._precip_type(env))

        # Wind
        wind = env.get("wind_speed_kmh")
        if wind is not None:
            weather.set("windSpeed", f"{wind:.1f}")
        wind_dir = env.get("wind_direction_deg")
        if wind_dir is not None:
            weather.set("windDirection", f"{wind_dir:.0f}")

        # Visibility
        vis = env.get("visibility_m")
        if vis is not None:
            weather.set("visibility", f"{vis:.0f}")

        # Friction
        friction = _compute_friction(
            temp,
            rain,
            env.get("snowfall_cm", 0),
            env.get("road_condition", "dry"),
        )
        weather.set("friction", f"{friction:.3f}")

        # Surface condition
        weather.set("surfaceCondition", env.get("road_condition", "dry"))

        # Cloud cover
        cloud = env.get("cloud_cover_pct")
        if cloud is not None:
            weather.set("cloudCover", f"{cloud:.0f}")

        # Solar radiation
        solar = env.get("solar_radiation_wm2", env.get("shortwave_radiation_wm2"))
        if solar is not None:
            weather.set("solarRadiation", f"{solar:.1f}")

        # Day/Night
        weather.set("isDay", str(env.get("is_day", True)).lower())

        # Weather condition string
        weather.set("condition", env.get("weather_condition", "clear_sky"))

        self._write_xml(root, path)

    # ─────────────────────────────────────────────────────────────
    # 2. Full environment XML
    # ─────────────────────────────────────────────────────────────
    def _generate_environment_xml(
        self,
        path: Path,
        env: Dict[str, Any],
        lat: float,
        lon: float,
        dt: datetime,
        prediction_source: str,
    ) -> None:
        """Generate comprehensive environment.xml with elevation and bbox."""
        root = etree.Element("environment")

        # ── Location — includes elevation and 1km x 1km bbox ─────────
        loc = etree.SubElement(root, "location")
        loc.set("latitude",    f"{lat:.6f}")
        loc.set("longitude",   f"{lon:.6f}")
        loc.set("elevation_m", f"{env.get('elevation_m', 0.0):.1f}")   # ALTITUDE

        bbox = env.get("bbox", {})
        if bbox:
            bb = etree.SubElement(loc, "boundingBox")
            bb.set("south",    str(bbox.get("south", "")))
            bb.set("west",     str(bbox.get("west",  "")))
            bb.set("north",    str(bbox.get("north", "")))
            bb.set("east",     str(bbox.get("east",  "")))
            bb.set("width_m",  "1000")
            bb.set("height_m", "1000")

        # ── Temporal ──────────────────────────────────────────────────
        temporal = etree.SubElement(root, "temporal")
        temporal.set("datetime", dt.isoformat())
        temporal.set("predictionSource", prediction_source)

        # Atmospheric
        atmo = etree.SubElement(root, "atmospheric")
        for key, attr in [
            ("temperature_c", "temperature"),
            ("surface_pressure_hpa", "pressure"),
            ("relative_humidity_pct", "humidity"),
            ("cloud_cover_pct", "cloudCover"),
            ("visibility_m", "visibility"),
        ]:
            val = env.get(key)
            if val is not None:
                atmo.set(attr, f"{val:.2f}")

        # Precipitation
        precip = etree.SubElement(root, "precipitation")
        precip.set("rain_mm", f"{env.get('rainfall_mm_1h', 0):.2f}")
        precip.set("snow_cm", f"{env.get('snowfall_cm', 0):.2f}")
        precip.set("type", self._precip_type(env))
        if env.get("precipitation_label"):
            precip.set("intensityClass", env["precipitation_label"])

        # Wind
        wind = etree.SubElement(root, "wind")
        ws = env.get("wind_speed_kmh")
        if ws is not None:
            wind.set("speed_kmh", f"{ws:.1f}")
        wd = env.get("wind_direction_deg")
        if wd is not None:
            wind.set("direction_deg", f"{wd:.0f}")

        # Solar
        solar = etree.SubElement(root, "solar")
        sr = env.get("solar_radiation_wm2", env.get("shortwave_radiation_wm2"))
        if sr is not None:
            solar.set("radiation_wm2", f"{sr:.1f}")
        slf = env.get("sun_load_factor")
        if slf is not None:
            solar.set("sunLoadFactor", f"{slf:.4f}")
        solar.set("isDay", str(env.get("is_day", True)).lower())

        # Road surface
        road = etree.SubElement(root, "roadSurface")
        road.set("condition", env.get("road_condition", "dry"))
        friction = _compute_friction(
            env.get("temperature_c"),
            env.get("rainfall_mm_1h", 0),
            env.get("snowfall_cm", 0),
            env.get("road_condition", "dry"),
        )
        road.set("friction", f"{friction:.3f}")

        self._write_xml(root, path)

    # ─────────────────────────────────────────────────────────────
    # 3. Vehicle types with weather adjustments
    # ─────────────────────────────────────────────────────────────
    def _generate_vehicle_types_xml(
        self, path: Path, env: Dict[str, Any],
    ) -> None:
        """Generate vehicle type modifications based on weather and elevation."""
        root = etree.Element("additional")

        road_cond = env.get("road_condition", "dry")
        vis       = env.get("visibility_m")
        wind      = env.get("wind_speed_kmh")
        elev      = env.get("elevation_m", 0.0) or 0.0

        speed_factor  = _weather_speed_factor(road_cond, vis, wind)
        imperfection  = _visibility_to_imperfection(vis)

        # High altitude reduces engine power -> lower max accel for heavy vehicles
        # > 2 000 m: 85% power, > 1 000 m: 93% power
        if elev > 2000:
            altitude_decel = 0.85
        elif elev > 1000:
            altitude_decel = 0.93
        else:
            altitude_decel = 1.0

        # Default passenger car
        vtype = etree.SubElement(root, "vType")
        vtype.set("id",          "weather_adjusted_car")
        vtype.set("vClass",      "passenger")
        vtype.set("speedFactor", f"{speed_factor:.3f}")
        vtype.set("sigma",       f"{imperfection:.2f}")
        vtype.set("minGap",      f"{max(2.5, 2.5 / speed_factor):.2f}")
        vtype.set("accel",       f"{round(2.6 * altitude_decel, 2)}")
        vtype.set("decel",       f"{round(4.5 * altitude_decel, 2)}")
        p = etree.SubElement(vtype, "param")
        p.set("key", "elevation_m"); p.set("value", str(elev))

        # Truck (more affected by wind and altitude)
        truck_speed = speed_factor
        if wind and wind > 40:
            truck_speed *= 0.90
        truck = etree.SubElement(root, "vType")
        truck.set("id",          "weather_adjusted_truck")
        truck.set("vClass",      "truck")
        truck.set("speedFactor", f"{truck_speed:.3f}")
        truck.set("sigma",       f"{min(imperfection + 0.05, 1.0):.2f}")
        truck.set("minGap",      f"{max(3.0, 3.0 / truck_speed):.2f}")
        truck.set("accel",       f"{round(0.8 * altitude_decel, 2)}")
        truck.set("decel",       f"{round(4.0 * altitude_decel, 2)}")
        p2 = etree.SubElement(truck, "param")
        p2.set("key", "elevation_m"); p2.set("value", str(elev))

        # Emergency vehicle (less affected)
        emer = etree.SubElement(root, "vType")
        emer.set("id",          "weather_adjusted_emergency")
        emer.set("vClass",      "emergency")
        emer.set("speedFactor", f"{min(1.0, speed_factor * 1.15):.3f}")
        emer.set("sigma",       f"{max(0.1, imperfection - 0.2):.2f}")
        emer.set("accel",       f"{round(3.5 * altitude_decel, 2)}")
        emer.set("decel",       f"{round(7.0 * altitude_decel, 2)}")

        self._write_xml(root, path)

    # ─────────────────────────────────────────────────────────────
    # 4. Traffic light adjustments  (BluFleet metadata sidecar)
    # ─────────────────────────────────────────────────────────────
    def _generate_tls_xml(self, path: Path, env: Dict[str, Any]) -> None:
        """
        Write weather-derived TLS timing parameters as a BluFleet metadata
        sidecar XML.  This file is NOT passed to SUMO as an additional-file
        (doing so would require a matching junction ID in the network).
        It is kept alongside the scenario for reference / post-processing.
        """
        road_cond = env.get("road_condition", "dry")
        vis       = env.get("visibility_m")

        # Compute recommended extra all-red clearance (seconds)
        extra_clearance = 0
        if road_cond in ("wet", "flooded"):
            extra_clearance = 2
        elif road_cond == "snowy":
            extra_clearance = 4
        if vis is not None and vis < 1000:
            extra_clearance += 2

        root = etree.Element("tlsWeatherAdjustment")
        root.set("schemaVersion", "1.0")
        root.append(etree.Comment(
            " BluFleet TLS weather parameters — reference only, not loaded by SUMO "
        ))

        cond_el = etree.SubElement(root, "conditions")
        cond_el.set("road_condition",  road_cond)
        cond_el.set("visibility_m",    str(vis) if vis is not None else "unlimited")
        cond_el.set("has_tls",         str(env.get("has_traffic_lights", False)))

        adj_el = etree.SubElement(root, "adjustments")
        adj_el.set("extra_all_red_s",  str(extra_clearance))
        adj_el.set("yellow_extension", str(max(0, extra_clearance - 1)))

        rec_el = etree.SubElement(root, "recommendedPhases")
        phases = [
            ("green_ns",  42,                     "GGGGrrrrGGGGrrrr"),
            ("yellow_ns", 3 + extra_clearance,    "yyyyrrrryyyyrrrr"),
            ("allred_1",  2 + extra_clearance,    "rrrrrrrrrrrrrrrr"),
            ("green_ew",  38,                     "rrrrGGGGrrrrGGGG"),
            ("yellow_ew", 3 + extra_clearance,    "rrrryyyyrrrryyyy"),
            ("allred_2",  2 + extra_clearance,    "rrrrrrrrrrrrrrrr"),
        ]
        for name, dur, state in phases:
            ph = etree.SubElement(rec_el, "phase")
            ph.set("name",     name)
            ph.set("duration", str(dur))
            ph.set("state",    state)

        self._write_xml(root, path)

    # ─────────────────────────────────────────────────────────────
    # 5. Scenario config
    # ─────────────────────────────────────────────────────────────
    def _generate_config_xml(
        self,
        path: Path,
        scenario_id: str,
        lat: float,
        lon: float,
        dt: datetime,
        prediction_source: str,
        files: List[str],
        env: Dict[str, Any],
    ) -> None:
        """Generate scenario metadata manifest (BluFleet canonical record)."""
        root = etree.Element("scenarioManifest")
        root.set("id",        scenario_id)
        root.set("version",   "2.0")
        root.set("generated", datetime.now().isoformat())

        meta = etree.SubElement(root, "metadata")
        meta.set("latitude",         f"{lat:.6f}")
        meta.set("longitude",        f"{lon:.6f}")
        meta.set("elevation_m",      str(env.get("elevation_m", 0.0)))  # ALTITUDE
        meta.set("datetime",         dt.isoformat())
        meta.set("predictionSource", prediction_source)
        meta.set("bbox_km",          "1x1")

        bbox = env.get("bbox", {})
        if bbox:
            bb = etree.SubElement(meta, "boundingBox")
            for k, v in bbox.items():
                bb.set(k, str(v))
            bb.set("width_m", "1000")
            bb.set("height_m", "1000")

        sources = etree.SubElement(root, "dataSources")
        for s in ["open-meteo", "open-elevation", "overpass-osm", "astronomical-model"]:
            etree.SubElement(sources, "source").set("name", s)

        files_el = etree.SubElement(root, "generatedFiles")
        for fp in files:
            f_el = etree.SubElement(files_el, "file")
            f_el.set("path", os.path.basename(fp))
            f_el.set("type", Path(fp).suffix)

        summary = etree.SubElement(root, "environmentSummary")
        for key in [
            "temperature_c", "rainfall_mm_1h", "wind_speed_kmh",
            "visibility_m", "road_condition", "weather_condition",
            "solar_radiation_wm2", "elevation_m", "is_day",
        ]:
            val = env.get(key)
            if val is not None:
                summary.set(key, str(val))

        self._write_xml(root, path)

    # ─────────────────────────────────────────────────────────────
    # 5. Routes via randomTrips.py
    # ─────────────────────────────────────────────────────────────
    def _generate_routes(self, net_path: Path, out_dir: Path) -> Optional[Path]:
        """
        Generate random vehicle routes using SUMO's randomTrips.py.
        Falls back to a minimal stub if the tool is not found.
        """
        rou_path = out_dir / "routes.rou.xml"

        candidates = [
            "/opt/homebrew/share/sumo/tools/randomTrips.py",
            "/usr/share/sumo/tools/randomTrips.py",
            "/usr/local/share/sumo/tools/randomTrips.py",
        ]
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            candidates.insert(0, f"{sumo_home}/tools/randomTrips.py")

        random_trips = next((c for c in candidates if Path(c).exists()), None)

        if not random_trips:
            logger.warning("randomTrips.py not found — writing fallback routes stub")
            root = etree.Element("routes")
            etree.SubElement(root, "comment").text = "Fallback — no randomTrips.py found"
            self._write_xml(root, rou_path)
            return rou_path

        cmd = [
            sys.executable, random_trips,
            "-n", str(net_path),
            "-r", str(rou_path),
            "-b", "0",
            "-e", "600",
            "--period", "3",
            "--validate",
            "--min-distance", "200",
            "--vehicle-class", "passenger",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and rou_path.exists():
                logger.info("Routes generated -> %s", rou_path)
                return rou_path
            logger.warning("randomTrips.py failed: %s", result.stderr[-400:])
        except Exception as exc:
            logger.warning("Route generation error: %s", exc)

        # Fallback stub
        root = etree.Element("routes")
        etree.SubElement(root, "comment").text = "Fallback — randomTrips.py failed"
        self._write_xml(root, rou_path)
        return rou_path

    # ─────────────────────────────────────────────────────────────
    # 6. Master SUMO .sumocfg
    # ─────────────────────────────────────────────────────────────
    def _generate_sumocfg(
        self,
        path: Path,
        scenario_id: str,
        lat: float,
        lon: float,
        dt: datetime,
        env: Dict[str, Any],
        net_path: Optional[Path],
        rou_path: Optional[Path],
        extra_files: List[Path],
    ) -> None:
        """
        Write scenario.sumocfg — the master SUMO config.
        Elevation is embedded in the <location> block as altitude_m.
        """
        elev    = env.get("elevation_m", 0.0) or 0.0
        bbox    = env.get("bbox", {})
        friction= _compute_friction(
            env.get("temperature_c"),
            env.get("rainfall_mm_1h", 0),
            env.get("snowfall_cm", 0),
            env.get("road_condition", "dry"),
        )
        vis_m   = env.get("visibility_m", 30000) or 30000

        root = etree.Element(
            "configuration",
            nsmap={"xsi": "http://www.w3.org/2001/XMLSchema-instance"},
        )
        root.set(
            "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
            "http://sumo.dlr.de/xsd/sumoConfiguration.xsd",
        )

        # ── input ─────────────────────────────────────────────────────
        inp = etree.SubElement(root, "input")
        if net_path and net_path.exists():
            etree.SubElement(inp, "net-file").set("value", net_path.name)
        if rou_path and rou_path.exists():
            etree.SubElement(inp, "route-files").set("value", rou_path.name)
        if extra_files:
            etree.SubElement(inp, "additional-files").set(
                "value", ",".join(p.name for p in extra_files if p.exists())
            )

        # ── time ──────────────────────────────────────────────────────
        time_el = etree.SubElement(root, "time")
        etree.SubElement(time_el, "begin").set("value", "0")
        etree.SubElement(time_el, "end").set(  "value", "600")
        etree.SubElement(time_el, "step-length").set("value", "0.1")

        # ── processing ────────────────────────────────────────────────
        proc = etree.SubElement(root, "processing")
        etree.SubElement(proc, "collision.action").set("value", "warn")
        etree.SubElement(proc, "time-to-teleport").set("value", "300")

        # ── routing ───────────────────────────────────────────────────
        rout = etree.SubElement(root, "routing")
        etree.SubElement(rout, "device.rerouting.probability").set("value", "0.8")
        etree.SubElement(rout, "device.rerouting.period").set("value", "60")

        # ── report ────────────────────────────────────────────────────
        rep = etree.SubElement(root, "report")
        etree.SubElement(rep, "verbose").set("value", "true")
        etree.SubElement(rep, "no-step-log").set("value", "false")

        # ── output ────────────────────────────────────────────────────
        out = etree.SubElement(root, "output")
        etree.SubElement(out, "fcd-output").set(    "value", f"{scenario_id}_fcd.xml")
        etree.SubElement(out, "summary-output").set("value", f"{scenario_id}_summary.xml")
        etree.SubElement(out, "tripinfo-output").set("value", f"{scenario_id}_tripinfo.xml")

        # ── location — ELEVATION EMBEDDED HERE ────────────────────────
        loc = etree.SubElement(root, "location")
        loc.set("altitude_m",   f"{elev:.2f}")          # ALTITUDE
        loc.set("lat_center",   f"{lat:.6f}")
        loc.set("lon_center",   f"{lon:.6f}")
        if bbox:
            loc.set("origBoundary",
                    f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}")

        # ── environment parameters ────────────────────────────────────
        envp = etree.SubElement(root, "environmentParameters")
        envp.set("scenario_id",       scenario_id)
        envp.set("temperature_c",     str(env.get("temperature_c", 0)))
        envp.set("rainfall_mm_1h",    str(env.get("rainfall_mm_1h", 0)))
        envp.set("wind_speed_kmh",    str(env.get("wind_speed_kmh",  0)))
        envp.set("visibility_m",      str(vis_m))
        envp.set("friction",          str(friction))
        envp.set("elevation_m",       str(elev))         # ALTITUDE
        envp.set("solar_radiation",   str(env.get("solar_radiation_wm2", 0)))
        envp.set("weather_condition", env.get("weather_condition", "clear_sky"))

        self._write_xml(root, path)
        logger.info("Written: %s", path)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _precip_type(env: Dict[str, Any]) -> str:
        snow = env.get("snowfall_cm", 0) or 0
        rain = env.get("rainfall_mm_1h", 0) or 0
        if snow > 0:
            return "snow"
        if rain > 0:
            return "rain"
        return "none"

    @staticmethod
    def _write_xml(root: etree._Element, path: Path) -> None:
        """Write pretty-printed XML to file."""
        xml_bytes = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="UTF-8")
        path.write_bytes(xml_bytes)
        logger.info("Written: %s", path)
