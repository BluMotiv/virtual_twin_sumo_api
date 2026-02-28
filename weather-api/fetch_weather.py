"""
fetch_weather.py
================
Standalone script — Open-Meteo weather extraction.
Lives in: BluFleet AI/weather-api/

✅  NO API KEY required — Open-Meteo is completely free.

Usage
-----
    # By place name (geocoded automatically)
    python fetch_weather.py --place "Hyderabad"

    # By lat/lon directly (skips geocoding)
    python fetch_weather.py --lat 17.4399 --lon 78.4983

    # Historical weather for a date range
    python fetch_weather.py --place "Hyderabad" --start 2026-01-01 --end 2026-01-07

    # Forecast for next 3 days
    python fetch_weather.py --lat 17.4399 --lon 78.4983 --days 3

    # Save output to JSON
    python fetch_weather.py --place "Hyderabad" --save

Run from this folder:
    cd "BluFleet AI/weather-api"
    python fetch_weather.py --place "Hyderabad"
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from collections import Counter

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from weather.open_meteo_client import OpenMeteoClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── SUMO scenario parameters we care about ─────────────────────────────────
SUMO_PARAMS = [
    "time",
    "temperature_c",
    "rainfall_mm_1h",
    "rainfall_mm_24h",
    "snowfall_cm",
    "wind_speed_kmh",
    "wind_direction_deg",
    "visibility_m",
    "relative_humidity_pct",
    "surface_pressure_hpa",
    "cloud_cover_pct",
    "weather_condition",
    "road_condition",
    "is_day",
]


def print_location_banner(geo: dict) -> None:
    """Print a clear box showing the resolved coordinates."""
    width = 54
    print("\n" + "═" * width)
    print("  📍  LOCATION RESOLVED")
    print("─" * width)
    print(f"  Name      : {geo.get('name', '—')}, {geo.get('country', '—')}")
    print(f"  Latitude  : {geo.get('lat', 0):.6f} °")
    print(f"  Longitude : {geo.get('lon', 0):.6f} °")
    print(f"  Elevation : {geo.get('elevation', 0):.0f} m")
    print(f"  Timezone  : {geo.get('timezone', 'UTC')}")
    print("═" * width + "\n")


def print_snapshot(snap: dict, title: str = "") -> None:
    width = 54
    print("\n" + "─" * width)
    if title:
        print(f"  {title}")
        print("─" * width)
    loc = snap.get("location", {})
    if loc:
        print(f"  📍 {loc.get('name')}, {loc.get('country')}  "
              f"(lat={loc.get('lat'):.4f}, lon={loc.get('lon'):.4f})")
        print("─" * width)

    for key in SUMO_PARAMS:
        val = snap.get(key)
        label = key.replace("_", " ").title().ljust(28)
        if val is None:
            print(f"  {label}: —")
        elif isinstance(val, float):
            print(f"  {label}: {val:.2f}")
        elif isinstance(val, bool):
            print(f"  {label}: {'yes' if val else 'no'}")
        else:
            print(f"  {label}: {val}")
    print("─" * width + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch SUMO-ready weather data from Open-Meteo (no API key).",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Location: place name OR explicit lat/lon ───────────────────────────
    loc_group = parser.add_mutually_exclusive_group(required=True)
    loc_group.add_argument("--place",
                           help="Location name, e.g. 'Kalgoorlie'")
    loc_group.add_argument("--lat", type=float,
                           help="Latitude in decimal degrees, e.g. -30.7461\n"
                                "(must be used together with --lon)")

    parser.add_argument("--lon", type=float,
                        help="Longitude in decimal degrees, e.g. 121.4742\n"
                             "(required when --lat is used)")

    # ── Time window ────────────────────────────────────────────────────────
    parser.add_argument("--days", type=int, default=1,
                        help="Forecast days (1–16, default: 1)")
    parser.add_argument("--start",
                        help="Historical start date YYYY-MM-DD (switches to archive API)")
    parser.add_argument("--end",
                        help="Historical end date   YYYY-MM-DD (defaults to --start)")

    # ── Display / output ───────────────────────────────────────────────────
    parser.add_argument("--hour", type=int, default=None,
                        help="Show only this hour index (0-based)")
    parser.add_argument("--save", action="store_true",
                        help="Save full result to data/raw/<name>_weather.json")

    args = parser.parse_args()

    # ── Validate lat/lon pairing ───────────────────────────────────────────
    if args.lat is not None and args.lon is None:
        parser.error("--lon is required when --lat is provided.")
    if args.lon is not None and args.lat is None:
        parser.error("--lat is required when --lon is provided.")

    client = OpenMeteoClient()

    # ── Resolve coordinates ────────────────────────────────────────────────
    if args.lat is not None:
        # Direct lat/lon — no geocoding needed
        geo = {
            "name":      f"({args.lat}, {args.lon})",
            "country":   "",
            "lat":       args.lat,
            "lon":       args.lon,
            "elevation": 0,
            "timezone":  "auto",
        }
        print(f"\n🗺   Using explicit coordinates: lat={args.lat}, lon={args.lon}")
    else:
        # Geocode from place name
        print(f"\n🔍  Geocoding '{args.place}' …")
        geo = client.geocode(args.place)

    # Always show the resolved location box
    print_location_banner(geo)

    safe_name = (
        args.place.replace(" ", "_").replace(",", "").lower()
        if args.place
        else f"lat{args.lat}_lon{args.lon}".replace("-", "m").replace(".", "p")
    )

    # ── Historical mode ────────────────────────────────────────────────────
    if args.start:
        end = args.end or args.start
        print(f"🌦  Fetching historical weather  {args.start} → {end} …")
        result = client.fetch_historical(
            lat=geo["lat"], lon=geo["lon"],
            start_date=args.start, end_date=end,
            timezone=geo.get("timezone", "auto"),
        )
        result["location"] = geo

    # ── Forecast mode ──────────────────────────────────────────────────────
    else:
        print(f"🌤  Fetching {args.days}-day forecast …")
        result = client.fetch_forecast(
            lat=geo["lat"],
            lon=geo["lon"],
            timezone=geo.get("timezone", "auto"),
            forecast_days=args.days,
        )
        result["location"] = geo

    snapshots = result.get("hourly_snapshots", [])
    location  = result.get("location", {})

    if not snapshots:
        print("❌  No weather data returned.")
        sys.exit(1)

    print(f"✅  Retrieved {len(snapshots)} hourly snapshots.\n")

    # ── Display ────────────────────────────────────────────────────────────
    if args.hour is not None:
        if args.hour >= len(snapshots):
            print(f"❌  --hour {args.hour} out of range (0–{len(snapshots)-1}).")
            sys.exit(1)
        snap = snapshots[args.hour]
        snap["location"] = location
        print_snapshot(snap, title=f"Hour [{args.hour}]  {snap['time']}")
    else:
        # Show first, current-hour, and last snapshots
        now_str = datetime.now().strftime("%Y-%m-%dT%H")
        shown   = set()
        for i, snap in enumerate(snapshots):
            is_now = snap["time"].startswith(now_str)
            if i == 0 or i == len(snapshots) - 1 or is_now:
                if i not in shown:
                    snap["location"] = location
                    label = "Current hour" if is_now else (
                        "First snapshot" if i == 0 else "Last snapshot"
                    )
                    print_snapshot(snap, title=f"{label}  [{i}]  {snap['time']}")
                    shown.add(i)

    # ── Summary table ──────────────────────────────────────────────────────
    temps  = [s["temperature_c"]  for s in snapshots if s["temperature_c"]  is not None]
    rains  = [s["rainfall_mm_1h"] for s in snapshots if s["rainfall_mm_1h"] is not None]
    winds  = [s["wind_speed_kmh"] for s in snapshots if s["wind_speed_kmh"] is not None]
    visib  = [s["visibility_m"]   for s in snapshots if s["visibility_m"]   is not None]

    print("📊  Summary across all snapshots:")
    if temps:
        print(f"    Temperature   : min={min(temps):.1f}°C  max={max(temps):.1f}°C  avg={sum(temps)/len(temps):.1f}°C")
    if rains:
        print(f"    Rain (1h)     : max={max(rains):.2f}mm  total={sum(rains):.2f}mm")
    if winds:
        print(f"    Wind speed    : max={max(winds):.1f} km/h")
    if visib:
        print(f"    Visibility    : min={min(visib):.0f}m  avg={sum(visib)/len(visib):.0f}m")

    ctr = Counter(s["road_condition"] for s in snapshots)
    print(f"    Road cond.    : {dict(ctr)}")

    # ── Save to JSON ───────────────────────────────────────────────────────
    if args.save:
        raw_dir  = os.path.join(os.path.dirname(__file__), "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        out_path = os.path.join(raw_dir, f"{safe_name}_weather.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\n💾  Saved → {out_path}")

    print("\n✅  Done. Weather data is ready for SUMO scenario injection.\n")


if __name__ == "__main__":
    main()
