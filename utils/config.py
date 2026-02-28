"""
utils/config.py
===============
Centralised application settings loaded from environment / .env file.

Uses pydantic-settings for validation and type coercion.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root (where main.py lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configurable knobs — override via env vars or .env."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"

    # ── Open-Meteo endpoints ────────────────────────────────────
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_geocode_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    open_meteo_timeout: int = 15

    # ── Elevation ───────────────────────────────────────────────
    open_elevation_url: str = "https://api.open-elevation.com/api/v1/lookup"

    # ── Output ──────────────────────────────────────────────────
    sumo_output_dir: str = "output/sumo"

    # ── ML Models ───────────────────────────────────────────────
    ml_models_dir: str = "mlmodels/MODELS"

    # ── Forecast decision window ────────────────────────────────
    forecast_window_days: int = 10

    # ── Cache TTL ───────────────────────────────────────────────
    cache_ttl: int = 3600

    # ── Derived helpers ─────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def ml_models_path(self) -> Path:
        p = Path(self.ml_models_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p

    @property
    def sumo_output_path(self) -> Path:
        p = Path(self.sumo_output_dir)
        return p if p.is_absolute() else _PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings object."""
    return Settings()
