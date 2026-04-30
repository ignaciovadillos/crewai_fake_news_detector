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
    local_appdata = state_root / "localappdata"
    roaming_appdata = state_root / "appdata"
    crewai_storage = state_root / "crewai_storage"

    data_home.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    local_appdata.mkdir(parents=True, exist_ok=True)
    roaming_appdata.mkdir(parents=True, exist_ok=True)
    crewai_storage.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_DATA_HOME"] = str(data_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_home)
    os.environ["LOCALAPPDATA"] = str(local_appdata)
    os.environ["APPDATA"] = str(roaming_appdata)
    os.environ["CREWAI_STORAGE_DIR"] = str(crewai_storage)
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
