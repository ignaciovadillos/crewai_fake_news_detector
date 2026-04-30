"""Deterministic contradiction detection between evidence cases."""

from __future__ import annotations

import re

from detector_fake_news.models import ContradictionReport, LegalCase

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


def detect_contradictions(
    supporting_case: LegalCase,
    opposing_case: LegalCase,
) -> ContradictionReport:
    """Detect meaningful conflict between supporting and opposing evidence."""
    contradiction_count = 0
    mixed_signal_count = 0
    notes: list[str] = []

    for support_item in supporting_case.results:
        matched_opposing = _best_match(support_item.claim, opposing_case)
        if matched_opposing is None:
            continue

        support_is_strong = (
            support_item.verdict == "SUPPORTED"
            and support_item.confidence >= 0.65
            and bool(support_item.source_urls)
        )
        opposing_is_strong = (
            matched_opposing.verdict == "CONTRADICTED"
            and matched_opposing.confidence >= 0.65
            and bool(matched_opposing.source_urls)
        )
        opposing_is_uncertain = matched_opposing.verdict in {"MIXED", "UNVERIFIABLE"}

        if support_is_strong and opposing_is_strong:
            contradiction_count += 1
            notes.append(
                f"Strong conflict on claim: {support_item.claim[:180]}"
            )
        elif support_is_strong and opposing_is_uncertain:
            mixed_signal_count += 1
            notes.append(
                f"Supported claim has unresolved opposing evidence: {support_item.claim[:180]}"
            )

    total_items = max(len(supporting_case.results), len(opposing_case.results), 1)
    score = min((contradiction_count * 0.5 + mixed_signal_count * 0.25) / total_items, 1.0)
    level = _level(score)
    if not notes:
        notes.append("No strong support-versus-contradiction conflicts were detected.")

    return ContradictionReport(
        score=round(score, 4),
        level=level,
        contradiction_count=contradiction_count,
        mixed_signal_count=mixed_signal_count,
        notes=notes[:6],
    )


def _best_match(claim: str, legal_case: LegalCase):
    claim_tokens = _tokenize(claim)
    best_score = 0.0
    best_item = None
    for item in legal_case.results:
        item_tokens = _tokenize(item.claim)
        if not claim_tokens or not item_tokens:
            continue
        overlap = claim_tokens.intersection(item_tokens)
        score = len(overlap) / max(len(claim_tokens.union(item_tokens)), 1)
        if score > best_score:
            best_score = score
            best_item = item
    return best_item if best_score >= 0.25 else None


def _tokenize(text: str) -> set[str]:
    return {
        word.lower().strip("'")
        for word in _WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }


def _level(score: float) -> str:
    if score >= 0.45:
        return "HIGH"
    if score >= 0.2:
        return "MEDIUM"
    return "LOW"
