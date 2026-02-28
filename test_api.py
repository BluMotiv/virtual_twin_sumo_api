"""
test_api.py
===========
Self-contained integration test for the SUMO Scenario Generation API.
Starts the server, runs tests, and shuts down.
"""

import asyncio
import json
import sys
import os

# Ensure sumoapi is on the path
sys.path.insert(0, os.path.dirname(__file__))

async def test():
    from main import app, lifespan
    from httpx import AsyncClient, ASGITransport

    # Trigger lifespan to load models
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # ── Test 1: Health ──
            print("=" * 60)
            print("TEST 1: GET /health")
            print("=" * 60)
            r = await client.get("/health")
            print(json.dumps(r.json(), indent=2))
            assert r.status_code == 200
            assert r.json()["models_count"] == 4
            print("✅ PASSED\n")

            # ── Test 2: Models ──
            print("=" * 60)
            print("TEST 2: GET /models")
            print("=" * 60)
            r = await client.get("/models")
            models = r.json()["models"]
            print(f"Loaded models: {list(models.keys())}")
            for name, info in models.items():
                print(f"  {name}: {info['model_type']} ({info['feature_count']} features)")
            assert len(models) == 4
            print("✅ PASSED\n")

            # ── Test 3: Forecast mode ──
            print("=" * 60)
            print("TEST 3: POST /generate-scenario (FORECAST mode)")
            print("=" * 60)
            r = await client.post("/generate-scenario", json={
                "latitude": 17.4399,
                "longitude": 78.4983,
                "date": "2026-03-05",
                "time": "14:30",
            }, timeout=30.0)
            data = r.json()
            print(json.dumps(data, indent=2))
            assert r.status_code == 200
            assert data["prediction_source"] == "forecast"
            assert len(data["sumo_files_generated"]) >= 4
            assert data["environment_features"]["temperature_c"] is not None
            print("✅ PASSED\n")

            # ── Test 4: ML model mode ──
            print("=" * 60)
            print("TEST 4: POST /generate-scenario (ML MODEL mode)")
            print("=" * 60)
            r = await client.post("/generate-scenario", json={
                "latitude": 17.4399,
                "longitude": 78.4983,
                "date": "2026-07-15",
                "time": "10:00",
            }, timeout=60.0)
            data = r.json()
            print(json.dumps(data, indent=2))
            assert r.status_code == 200
            assert data["prediction_source"] == "ml_model"
            assert len(data["models_used"]) >= 3
            assert data["environment_features"]["temperature_c"] is not None
            print("✅ PASSED\n")

            # ── Test 5: Validation ──
            print("=" * 60)
            print("TEST 5: POST /generate-scenario (VALIDATION — bad date)")
            print("=" * 60)
            r = await client.post("/generate-scenario", json={
                "latitude": 17.4399,
                "longitude": 78.4983,
                "date": "not-a-date",
                "time": "10:00",
            })
            assert r.status_code == 422
            print(f"Status: {r.status_code} (expected 422)")
            print("✅ PASSED\n")

            print("=" * 60)
            print("ALL TESTS PASSED ✅")
            print("=" * 60)
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    asyncio.run(test())
