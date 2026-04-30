"""Persistent local run history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from detector_fake_news.models import PipelineReport


def record_single_run(
    report: PipelineReport,
    *,
    evidence_mode: str,
    model_name: str,
    analysis_depth: str,
    use_memory: bool,
) -> None:
    """Persist a compact single-article run history entry."""
    record = {
        "created_at": _now(),
        "run_type": "single",
        "title": report.title,
        "evidence_mode": evidence_mode,
        "model_name": model_name or "default",
        "analysis_depth": analysis_depth,
        "memory_enabled": use_memory,
        "label": report.final_verdict.label,
        "truth_score": round(report.final_verdict.truth_score, 4),
        "confidence": round(report.final_verdict.confidence, 4),
        "evidence_quality": (
            round(report.evidence_quality.score, 4)
            if report.evidence_quality
            else None
        ),
        "contradiction_level": (
            report.contradiction_report.level
            if report.contradiction_report
            else ""
        ),
        "summary": report.final_verdict.summary[:500],
    }
    _append_history(record)


def record_batch_run(
    rows: list[dict[str, str]],
    *,
    evidence_mode: str,
    model_name: str,
    analysis_depth: str,
    use_memory: bool,
) -> None:
    """Persist a compact batch run history entry."""
    completed = [row for row in rows if row.get("status") == "OK"]
    errors = [row for row in rows if row.get("status") == "ERROR"]
    skipped = [row for row in rows if row.get("status") == "SKIPPED"]
    correct = [row for row in completed if row.get("expected_match") == "CORRECT"]
    scored = [row for row in completed if row.get("expected_match") in {"CORRECT", "INCORRECT"}]
    durations = [
        float(row["duration_seconds"])
        for row in rows
        if row.get("duration_seconds")
    ]

    record = {
        "created_at": _now(),
        "run_type": "batch",
        "title": f"Batch run ({len(rows)} row(s))",
        "batch_run_id": rows[0].get("batch_run_id", "") if rows else "",
        "evidence_mode": evidence_mode,
        "model_name": model_name or "default",
        "analysis_depth": analysis_depth,
        "memory_enabled": use_memory,
        "rows": len(rows),
        "completed": len(completed),
        "errors": len(errors),
        "skipped": len(skipped),
        "slow_rows": sum(1 for row in rows if row.get("is_slow") == "True"),
        "avg_duration_seconds": (
            round(sum(durations) / len(durations), 2)
            if durations
            else None
        ),
        "max_duration_seconds": round(max(durations), 2) if durations else None,
        "accuracy": round(len(correct) / len(scored), 4) if scored else None,
        "label_counts": _label_counts(completed),
    }
    _append_history(record)


def recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent run history entries, newest first."""
    records = _read_history()
    return records[-limit:][::-1]


def clear_run_history() -> None:
    """Delete local run history."""
    path = _history_path()
    if path.exists():
        path.unlink()


def _append_history(record: dict[str, Any]) -> None:
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _read_history() -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _history_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".runtime" / "run_history.jsonl"


def _label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("label", "")
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
