"""Lightweight helper functions shared by the Streamlit UI and tests."""

from __future__ import annotations

import csv
import io


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    """Serialize heterogeneous row dictionaries while preserving all columns."""
    if not rows:
        return ""
    buffer = io.StringIO()
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def expected_match(expected_label: str, predicted_label: str) -> str:
    if not expected_label:
        return "NO EXPECTED LABEL"
    normalized_expected = normalize_expected_label(expected_label)
    if predicted_label in {"ERROR", "SKIPPED"}:
        return "NOT SCORED"
    if not predicted_label:
        return "NOT SCORED"
    return "CORRECT" if normalized_expected == predicted_label else "INCORRECT"


def baseline_agreement(agent_label: str, baseline_label: str) -> str:
    if agent_label not in {"REAL", "FAKE"}:
        return "N/A"
    return "AGREE" if agent_label == baseline_label else "DISAGREE"


def normalize_expected_label(value: str) -> str:
    normalized = value.strip().upper()
    if normalized == "TRUE":
        return "REAL"
    if normalized == "FALSE":
        return "FAKE"
    return normalized
