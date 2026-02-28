#!/usr/bin/env python3
"""
run_scenario.py
===============
Interactive CLI for generating and launching a SUMO scenario.

Usage
-----
    python3 run_scenario.py                          # fully interactive
    python3 run_scenario.py 12.9784 77.6408          # lat/lon provided
    python3 run_scenario.py 12.9784 77.6408 2026-03-24 12:00   # all args

The script:
  1. Collects lat / lon / date / time (from args or prompts)
  2. Calls the running API at http://127.0.0.1:8000/generate-scenario
  3. Pretty-prints the full environment summary
  4. Asks whether you want to open the simulation in sumo-gui
  5. Detects macOS / Windows / Linux and launches sumo-gui automatically
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

# ── Optional: use requests if available, else fall back to urllib ──
try:
    import requests as _requests

    def _post(url: str, payload: dict) -> dict:
        r = _requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

except ImportError:
    import json
    import urllib.error
    import urllib.request

    def _post(url: str, payload: dict) -> dict:  # type: ignore[misc]
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


API_URL = "http://127.0.0.1:8000/generate-scenario"

# ─────────────────────────────────────────────────────────────────
# ANSI colours (disabled on Windows unless ANSI is supported)
# ─────────────────────────────────────────────────────────────────
_WIN = platform.system() == "Windows"
_ANSI = not _WIN or shutil.which("wt")  # Windows Terminal supports ANSI

def _c(code: str, text: str) -> str:
    if not _ANSI:
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD  = lambda t: _c("1",     t)
CYAN  = lambda t: _c("1;36",  t)
GREEN = lambda t: _c("1;32",  t)
YELLOW= lambda t: _c("1;33",  t)
RED   = lambda t: _c("1;31",  t)
DIM   = lambda t: _c("2",     t)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _ask(prompt: str, default: str = "") -> str:
    """Prompt with optional default."""
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {CYAN('?')} {prompt}{hint}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(0)
    return val if val else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"\n  {CYAN('?')} {prompt} [{hint}]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(0)
    if val in ("y", "yes"):
        return True
    if val in ("n", "no"):
        return False
    return default


def _row(key: str, val, unit: str = "", width: int = 32) -> None:
    """Print a single labelled data row."""
    if val is None:
        return
    val_str = f"{val} {unit}".strip() if unit else str(val)
    print(f"    {DIM(key + ':'):<{width}} {GREEN(val_str)}")


def _section(title: str) -> None:
    print(f"\n  {BOLD(title)}")


def _divider(char: str = "─", width: int = 50) -> None:
    print(f"  {DIM(char * width)}")


def _print_full_summary(
    env: dict,
    src: str,
    models: list,
    bbox: dict,
    sid: str,
    lat: float,
    lon: float,
    date_str: str,
    time_str: str,
) -> None:
    """
    Print a structured, sourced breakdown of everything the API produced.
    Sections:
      1. Request
      2. Open-Source Data Fetched   ← what came from external APIs
      3. ML Model Outputs           ← what the ML models predicted
      4. Derived / Computed         ← calculated internally (astronomy, logic)
      5. SUMO Network               ← OSM + netconvert output
    """

    # ── Header ────────────────────────────────────────────────────
    print(BOLD("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))
    print(f"  {BOLD('Scenario ID :')}  {GREEN(sid)}")
    print(f"  {BOLD('Prediction  :')}  {YELLOW(src)}")
    print(BOLD("  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

    # ── 1. Request ────────────────────────────────────────────────
    _section("🔷  1. Request")
    _divider()
    _row("Latitude",   lat,      "°")
    _row("Longitude",  lon,      "°")
    _row("Date",       date_str)
    _row("Time",       time_str)
    _row("Source mode", src)

    # ── 2. Open-Source Data Fetched ───────────────────────────────
    _section("🌐  2. Open-Source Data Fetched")
    _divider()

    # 2a. Open-Elevation
    print(f"\n    {CYAN('▸ Open-Elevation API')}  {DIM('(api.open-elevation.com)')}")
    _row("  Elevation",          env.get("elevation_m"),           "m asl")

    # 2b. Overpass / OpenStreetMap
    print(f"\n    {CYAN('▸ Overpass / OpenStreetMap')}")
    _row("  Road name",          env.get("road_name"))
    _row("  Road type",          env.get("road_type"))
    _row("  Surface",            env.get("road_surface"))
    _row("  Speed limit",        env.get("speed_limit"),           "km/h")
    _row("  Lanes",              env.get("lanes"))
    _row("  Traffic lights",     env.get("has_traffic_lights"))
    _row("  Road count (1km²)",  env.get("road_count"),            "roads")

    # 2c. Open-Meteo (only shown when source = forecast)
    if src == "forecast":
        print(f"\n    {CYAN('▸ Open-Meteo Forecast API')}  {DIM('(api.open-meteo.com)')}")
        _row("  Temperature",        env.get("temperature_c"),         "°C")
        _row("  Rainfall",           env.get("rainfall_mm_1h"),        "mm/h")
        _row("  Snowfall",           env.get("snowfall_cm"),            "cm")
        _row("  Wind speed",         env.get("wind_speed_kmh"),         "km/h")
        _row("  Wind direction",     env.get("wind_direction_deg"),     "°")
        _row("  Humidity",           env.get("relative_humidity_pct"),  "%")
        _row("  Visibility",         env.get("visibility_m"),           "m")
        _row("  Surface pressure",   env.get("surface_pressure_hpa"),   "hPa")
        _row("  Cloud cover",        env.get("cloud_cover_pct"),        "%")
        _row("  Weather condition",  env.get("weather_condition"))
        _row("  Is day",             env.get("is_day"))
        _row("  Shortwave radiation",env.get("shortwave_radiation"),    "W/m²")
    else:
        print(f"\n    {CYAN('▸ Open-Meteo Archive API')}  {DIM('(archive-api.open-meteo.com)')}")
        print(f"      {DIM('Target date is >7 days ahead — no direct forecast available.')}")
        print(f"      {DIM('Historical lags (up to yesterday) fetched for ML feature input:')}")
        print(f"      {DIM('  • 30-day daily mean temperatures    → T2M_LAG_1, _7, _30')}")
        print(f"      {DIM('  • 14-day daily precip totals        → lag_1, _3, _7, _14, roll sums')}")
        print(f"      {DIM('  • 24h hourly wind speeds + dir      → wind_lag_1h, mean_3h, max_24h')}")
        print(f"      {DIM('  • 24h hourly solar radiation        → sunload_lag_1h, mean_3h, max_24h')}")

    # 2d. 1km bbox
    if bbox:
        print(f"\n    {CYAN('▸ Bounding Box')}  {DIM('(computed from lat/lon, 1 km × 1 km)')}")
        _row("  South", bbox.get("south"), "°")
        _row("  West",  bbox.get("west"),  "°")
        _row("  North", bbox.get("north"), "°")
        _row("  East",  bbox.get("east"),  "°")

    # ── 3. ML Model Outputs ───────────────────────────────────────
    _section("🤖  3. ML Model Outputs")
    _divider()
    if src == "ml_model" and models:
        print(f"    {DIM('Models used: ')}{CYAN(', '.join(models))}")
        print()
        _row("  Temperature",         env.get("temperature_c"),        "°C",   34)
        _row("  Wind speed",          env.get("wind_speed_kmh"),        "km/h", 34)
        _row("  Solar radiation",     env.get("solar_radiation_wm2"),   "W/m²", 34)
        _row("  Precipitation class", env.get("precipitation_class"),   "",     34)
        _row("  Precipitation label", env.get("precipitation_label"),   "",     34)
    else:
        print(f"    {DIM('ML models not used — forecast data was available.')}")

    # ── 4. Derived / Computed ─────────────────────────────────────
    _section("🔬  4. Derived / Computed Internally")
    _divider()
    print(f"\n    {CYAN('▸ Astronomical Sun Model')}")
    _row("  Solar radiation",    env.get("solar_radiation_wm2"),   "W/m²")
    _row("  Sun load factor",    env.get("sun_load_factor"),        "  (0–1)")
    _row("  Is day",             env.get("is_day"))

    print(f"\n    {CYAN('▸ Road Condition Logic')}")
    _row("  Road condition",     env.get("road_condition"))
    _row("  Weather condition",  env.get("weather_condition"))

    # ── 5. SUMO Network ───────────────────────────────────────────
    _section("🗺   5. SUMO Network  (OSM → netconvert)")
    _divider()
    print(f"    {DIM('Source: Overpass OSM download → netconvert with elevation z-offset')}")
    if bbox:
        print(f"    {DIM('Coverage: 1 km × 1 km around the coordinate')}")


def _find_sumo_gui() -> str | None:
    """
    Locate the sumo-gui binary.
    Order: PATH → common install prefixes per OS.
    """
    # 1. Already on PATH
    found = shutil.which("sumo-gui")
    if found:
        return found

    system = platform.system()

    if system == "Darwin":          # macOS (Homebrew)
        candidates = [
            "/opt/homebrew/bin/sumo-gui",
            "/usr/local/bin/sumo-gui",
            "/Applications/sumo-gui.app/Contents/MacOS/sumo-gui",
        ]
    elif system == "Windows":
        candidates = [
            r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
            r"C:\sumo\bin\sumo-gui.exe",
        ]
    else:                           # Linux
        candidates = [
            "/usr/bin/sumo-gui",
            "/usr/local/bin/sumo-gui",
            "/snap/bin/sumo-gui",
        ]

    for c in candidates:
        if Path(c).exists():
            return c

    return None


def _launch_sumo(sumocfg: str) -> None:
    """Launch sumo-gui with the generated .sumocfg."""
    binary = _find_sumo_gui()
    system = platform.system()

    if binary is None:
        print(RED("\n  ✗  sumo-gui not found on this system."))
        print(DIM(f"     Install SUMO and add it to PATH, then run manually:"))
        print(f"     sumo-gui -c \"{sumocfg}\"")
        return

    print(f"\n  {GREEN('▶')}  Launching sumo-gui on {BOLD(system)} …")
    print(DIM(f"     {binary} -c {sumocfg}"))

    try:
        if system == "Windows":
            # On Windows open in a new console window so it doesn't block
            subprocess.Popen(
                [binary, "-c", sumocfg],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        elif system == "Darwin":
            # macOS: open detached from terminal
            subprocess.Popen(
                [binary, "-c", sumocfg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            # Linux
            subprocess.Popen(
                [binary, "-c", sumocfg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        print(f"  {GREEN('✓')}  sumo-gui started — the simulation window should appear shortly.")
    except Exception as exc:
        print(RED(f"\n  ✗  Failed to launch sumo-gui: {exc}"))
        print(DIM(f"     Try running manually:  sumo-gui -c \"{sumocfg}\""))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]

    print()
    print(BOLD(CYAN("  ╔══════════════════════════════════════════════╗")))
    print(BOLD(CYAN("  ║   BluFleet AI — SUMO Scenario Generator      ║")))
    print(BOLD(CYAN("  ╚══════════════════════════════════════════════╝")))
    print()

    # ── 1. Collect inputs ──────────────────────────────────────────
    today = date.today().isoformat()

    if len(args) >= 1:
        lat_str = args[0]
    else:
        lat_str = _ask("Latitude  (decimal, e.g. 12.9784)", "12.9784")

    if len(args) >= 2:
        lon_str = args[1]
    else:
        lon_str = _ask("Longitude (decimal, e.g. 77.6408)", "77.6408")

    if len(args) >= 3:
        date_str = args[2]
    else:
        date_str = _ask("Date      (YYYY-MM-DD)", today)

    if len(args) >= 4:
        time_str = args[3]
    else:
        time_str = _ask("Time      (HH:MM, 24h)", "12:00")

    try:
        latitude  = float(lat_str)
        longitude = float(lon_str)
    except ValueError:
        print(RED("  ✗  Invalid latitude/longitude. Must be numbers."))
        sys.exit(1)

    payload = {
        "latitude":  latitude,
        "longitude": longitude,
        "date":      date_str,
        "time":      time_str,
    }

    # ── 2. Call the API ────────────────────────────────────────────
    print()
    print(f"  {DIM('Generating scenario …')}")
    print(f"  {DIM(f'  lat={latitude}  lon={longitude}  date={date_str}  time={time_str}')}")
    print()

    try:
        result = _post(API_URL, payload)
    except Exception as exc:
        print(RED(f"  ✗  API call failed: {exc}"))
        print(DIM("     Is the API server running?  →  python3 -m uvicorn main:app --port 8000"))
        sys.exit(1)

    # ── 3. Print summary ───────────────────────────────────────────
    sid    = result.get("scenario_id", "—")
    src    = result.get("prediction_source", "—")
    files  = result.get("sumo_files_generated", [])
    bbox   = result.get("bbox_1km", {})
    netf   = result.get("network_file")
    runcmd = result.get("run_command", "")
    env    = result.get("environment_features", {})
    models = result.get("models_used", [])

    _print_full_summary(
        env=env, src=src, models=models, bbox=bbox,
        sid=sid, lat=latitude, lon=longitude,
        date_str=date_str, time_str=time_str,
    )

    # Files generated
    print(f"\n  {BOLD('📁  6. Generated SUMO Files')}  ({len(files)} total)")
    print(f"  {DIM('─' * 50)}")
    for f in files:
        fname = Path(f).name
        size  = ""
        try:
            size = f"  {DIM(str(round(Path(f).stat().st_size / 1024, 1)) + ' KB')}"
        except OSError:
            pass
        print(f"    {DIM('•')} {fname}{size}")

    if netf:
        print(f"\n    {DIM('Network path:')}")
        print(f"    {DIM(netf)}")

    print(BOLD("\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"))

    # ── 4. Ask to open sumo-gui ────────────────────────────────────
    if not runcmd:
        print(YELLOW("  ⚠  No .sumocfg was generated — cannot launch sumo-gui."))
        return

    # Extract the .sumocfg path from the run_command
    # run_command is like: "sumo-gui -c /path/to/scenario.sumocfg"
    parts = runcmd.split(" -c ", 1)
    sumocfg_path = parts[1].strip() if len(parts) == 2 else ""

    open_sim = _ask_yn("Open this scenario in sumo-gui?", default=True)

    if not open_sim:
        print(f"\n  {DIM('Run manually when ready:')}")
        print(f"    {CYAN(runcmd)}")
        print()
        return

    # Verify .sumocfg exists
    if sumocfg_path and not Path(sumocfg_path).exists():
        print(YELLOW(f"\n  ⚠  .sumocfg not found at expected path:"))
        print(f"     {sumocfg_path}")
        manual = _ask("Enter the correct path to .sumocfg (or press Enter to skip)", "")
        if manual:
            sumocfg_path = manual
        else:
            print(DIM("  Skipping launch."))
            return

    _launch_sumo(sumocfg_path)
    print()


if __name__ == "__main__":
    main()
