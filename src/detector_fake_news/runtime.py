"""Runtime environment configuration for local CrewAI execution."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment() -> None:
    """Point local app data to writable project paths."""
    project_root = Path(__file__).resolve().parents[2]
    state_root = project_root / ".runtime"
    data_home = state_root / "xdg_data"
    cache_home = state_root / "xdg_cache"

    data_home.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_DATA_HOME", str(data_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_home))
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
