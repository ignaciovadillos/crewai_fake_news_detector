"""Diagnostics helpers for batch runs."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_batch_run_id() -> str:
    """Create a readable run id that is safe for filenames."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"batch-{timestamp}-{uuid4().hex[:8]}"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_log_filename(batch_run_id: str) -> str:
    return f"{batch_run_id}-diagnostics.csv"


def batch_log_path(batch_run_id: str) -> Path:
    return _diagnostics_root() / f"{batch_run_id}.jsonl"


def describe_exception(exc: Exception | None) -> dict[str, str]:
    if exc is None:
        return {"error_type": "", "error": "", "traceback": ""}
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def append_batch_row_log(batch_run_id: str, entry: dict[str, Any]) -> None:
    """Append one row-level diagnostic event without interrupting the UI."""
    path = batch_log_path(batch_run_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        return


def _diagnostics_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".runtime" / "batch_diagnostics"
