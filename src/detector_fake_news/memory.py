"""Lightweight local memory for prior article analyses."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from detector_fake_news.models import PipelineReport

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")
_STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "both", "but", "can", "could",
    "did", "does", "for", "from", "had", "has", "have", "into", "its",
    "more", "not", "now", "off", "one", "only", "over", "said", "say",
    "says", "she", "should", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "too",
    "under", "was", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "you", "your",
}


def build_memory_context(
    *,
    title: str,
    article_text: str,
    evidence_mode: str,
    limit: int = 3,
) -> str:
    """Return a compact summary of similar previously analyzed articles."""
    query_tokens = _tokenize(f"{title} {article_text}")
    memories = _read_memories()
    scored: list[tuple[float, dict[str, Any]]] = []

    for memory in memories:
        if memory.get("evidence_mode") != evidence_mode:
            continue
        memory_tokens = set(memory.get("tokens", []))
        if not memory_tokens:
            continue
        overlap = query_tokens.intersection(memory_tokens)
        if not overlap:
            continue
        score = len(overlap) / max(len(query_tokens), 1)
        scored.append((score, memory | {"matched_terms": sorted(overlap)[:10]}))

    scored.sort(key=lambda item: (item[0], item[1].get("created_at", "")), reverse=True)
    selected = scored[:limit]
    if not selected:
        return "No similar prior analyses found in local memory."

    lines = [
        "Similar prior analyses from local project memory:",
    ]
    for score, memory in selected:
        lines.append(
            "- "
            f"Title: {memory.get('title', 'Untitled')} | "
            f"Prior label: {memory.get('label', 'UNKNOWN')} | "
            f"Truth score: {memory.get('truth_score', 'N/A')} | "
            f"Evidence mode: {memory.get('evidence_mode', 'unknown')} | "
            f"Similarity: {score:.2f} | "
            f"Matched terms: {', '.join(memory.get('matched_terms', []))} | "
            f"Prior summary: {memory.get('summary', '')}"
        )
    return "\n".join(lines)


def remember_analysis(report: PipelineReport, *, evidence_mode: str) -> None:
    """Persist a compact memory record for future runs."""
    memory_path = _memory_path()
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "id": _article_hash(report.title, report.article_text, evidence_mode),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": report.title,
        "evidence_mode": evidence_mode,
        "label": report.final_verdict.label,
        "truth_score": round(report.final_verdict.truth_score, 4),
        "confidence": round(report.final_verdict.confidence, 4),
        "bias_score": round(report.bias_report.bias_score, 4),
        "evidence_quality": (
            round(report.evidence_quality.score, 4)
            if report.evidence_quality
            else None
        ),
        "evidence_grade": report.evidence_quality.grade if report.evidence_quality else "",
        "contradiction_score": (
            round(report.contradiction_report.score, 4)
            if report.contradiction_report
            else None
        ),
        "contradiction_level": (
            report.contradiction_report.level
            if report.contradiction_report
            else ""
        ),
        "tone": report.bias_report.tone,
        "summary": report.final_verdict.summary[:600],
        "claims": report.claims.claims,
        "tokens": sorted(_tokenize(f"{report.title} {report.article_text}"))[:400],
    }

    memories = _read_memories()
    filtered = [memory for memory in memories if memory.get("id") != record["id"]]
    filtered.append(record)
    _write_memories(filtered[-500:])


def clear_analysis_memory() -> None:
    """Delete all local analysis memory records."""
    path = _memory_path()
    if path.exists():
        path.unlink()


def _read_memories() -> list[dict[str, Any]]:
    path = _memory_path()
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


def _write_memories(records: list[dict[str, Any]]) -> None:
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def _memory_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / ".runtime" / "analysis_memory.jsonl"


def _article_hash(title: str, article_text: str, evidence_mode: str) -> str:
    payload = "\n".join([title.strip(), article_text.strip(), evidence_mode.strip()])
    return sha256(payload.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> set[str]:
    return {
        word.lower().strip("'")
        for word in _WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
