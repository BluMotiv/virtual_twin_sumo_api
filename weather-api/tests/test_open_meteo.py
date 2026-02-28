"""
test_open_meteo.py
Tests for OpenMeteoClient and weather parsing logic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weather.open_meteo_client import (
    OpenMeteoClient,
    WMO_CODES,
    _wmo_to_road_condition,
    _safe_float,
)


# ---------------------------------------------------------------------------
# WMO helpers
# ---------------------------------------------------------------------------
class TestWmoHelpers:
    def test_clear_sky_is_dry(self):
        assert _wmo_to_road_condition(0) == "dry"

    def test_heavy_rain_is_flooded(self):
        assert _wmo_to_road_condition(65) == "flooded"

    def test_snow_is_snowy(self):
        assert _wmo_to_road_condition(73) == "snowy"

    def test_drizzle_is_wet(self):
        assert _wmo_to_road_condition(51) == "wet"

    def test_wmo_codes_not_empty(self):
        assert len(WMO_CODES) > 10

    def test_known_code_resolves(self):
        assert WMO_CODES[0] == "clear_sky"
        assert WMO_CODES[95] == "thunderstorm"


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------
class TestSafeFloat:
    def test_normal_value(self):
        assert _safe_float({"temp": [22.5, 23.0]}, "temp", 0) == 22.5

    def test_none_value(self):
        assert _safe_float({"temp": [None, 23.0]}, "temp", 0) is None

    def test_missing_key(self):
        assert _safe_float({}, "temp", 0) is None

    def test_out_of_range_index(self):
        assert _safe_float({"temp": [22.5]}, "temp", 5) is None


# ---------------------------------------------------------------------------
# OpenMeteoClient — parse_hourly (no network call)
# ---------------------------------------------------------------------------
MOCK_HOURLY_RESPONSE = {
    "latitude":  17.384,
    "longitude": 78.456,
    "timezone":  "Asia/Kolkata",
    "elevation": 515,
    "hourly": {
        "time":                ["2026-02-27T00:00", "2026-02-27T01:00"],
        "temperature_2m":      [23.3,  22.8],
        "relative_humidity_2m":[68.0,  70.0],
        "precipitation":       [0.0,   0.5],
        "rain":                [0.0,   0.5],
        "snowfall":            [0.0,   0.0],
        "snow_depth":          [0.0,   0.0],
        "weather_code":        [2,     61],
        "surface_pressure":    [951.9, 951.5],
        "cloud_cover":         [66.0,  80.0],
        "wind_speed_10m":      [4.1,   6.2],
        "wind_direction_10m":  [135.0, 140.0],
        "visibility":          [24140, 18000],
        "is_day":              [0,     0],
    },
}


class TestParseHourly:
    def setup_method(self):
        self.client = OpenMeteoClient()
        self.result = self.client._parse_hourly(MOCK_HOURLY_RESPONSE)

    def test_returns_two_snapshots(self):
        assert len(self.result["hourly_snapshots"]) == 2

    def test_temperature_parsed(self):
        assert self.result["hourly_snapshots"][0]["temperature_c"] == 23.3

    def test_rainfall_1h(self):
        assert self.result["hourly_snapshots"][1]["rainfall_mm_1h"] == 0.5

    def test_rainfall_24h_rolling(self):
        # hour 1 rolling = sum of hour 0 + hour 1 = 0.0 + 0.5
        assert self.result["hourly_snapshots"][1]["rainfall_mm_24h"] == pytest.approx(0.5)

    def test_road_condition_dry(self):
        assert self.result["hourly_snapshots"][0]["road_condition"] == "dry"

    def test_road_condition_wet_on_rain(self):
        assert self.result["hourly_snapshots"][1]["road_condition"] == "wet"

    def test_weather_condition_string(self):
        assert self.result["hourly_snapshots"][0]["weather_condition"] == "partly_cloudy"

    def test_is_day_false(self):
        assert self.result["hourly_snapshots"][0]["is_day"] is False

    def test_units_present(self):
        assert "temperature_c" in self.result["units"]

    def test_meta_lat_lon(self):
        assert self.result["meta"]["latitude"] == 17.384
        assert self.result["meta"]["longitude"] == 78.456


# ---------------------------------------------------------------------------
# OpenMeteoClient — live integration test (skipped if offline)
# ---------------------------------------------------------------------------
class TestLiveIntegration:
    @pytest.mark.integration
    def test_geocode_hyderabad(self):
        client = OpenMeteoClient()
        geo = client.geocode("Hyderabad")
        assert 17.0 < geo["lat"] < 18.0
        assert 78.0 < geo["lon"] < 79.0
        assert geo["country"] == "India"

    @pytest.mark.integration
    def test_fetch_forecast_returns_snapshots(self):
        client = OpenMeteoClient()
        result = client.fetch_forecast(lat=17.384, lon=78.456)
        assert len(result["hourly_snapshots"]) == 24
        snap = result["hourly_snapshots"][0]
        assert "temperature_c"    in snap
        assert "rainfall_mm_1h"   in snap
        assert "road_condition"   in snap
        assert "weather_condition" in snap
