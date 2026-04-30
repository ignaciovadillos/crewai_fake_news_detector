"""Evidence quality heuristics for fact-checking outputs."""

from __future__ import annotations

from urllib.parse import urlparse

from detector_fake_news.models import EvidenceQualityReport, LegalCase

_HIGH_QUALITY_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "cdc.gov",
    "congress.gov",
    "factcheck.org",
    "factcheck.afp.com",
    "fda.gov",
    "fullfact.org",
    "gov.uk",
    "leadstories.com",
    "nasa.gov",
    "nih.gov",
    "politifact.com",
    "reuters.com",
    "snopes.com",
    "who.int",
}

_MEDIUM_QUALITY_DOMAINS = {
    "en.wikipedia.org",
    "wikipedia.org",
}


def score_evidence_quality(*cases: LegalCase) -> EvidenceQualityReport:
    """Score evidence quality from source URLs and unresolved claim verdicts."""
    urls = [
        url
        for case in cases
        for item in case.results
        for url in item.source_urls
        if url
    ]
    unique_urls = sorted(set(urls))
    online_sources = [url for url in unique_urls if not url.startswith("offline://")]
    offline_sources = [url for url in unique_urls if url.startswith("offline://")]
    unsupported_claims = sum(
        1
        for case in cases
        for item in case.results
        if item.verdict in {"UNVERIFIABLE", "MIXED"} or not item.source_urls
    )
    total_claims = sum(len(case.results) for case in cases)

    source_component = min(len(unique_urls) / max(total_claims, 1), 1.0) * 0.35
    source_type_component = _source_type_score(online_sources, offline_sources) * 0.35
    resolution_component = (
        1.0 - min(unsupported_claims / max(total_claims, 1), 1.0)
    ) * 0.30
    score = round(source_component + source_type_component + resolution_component, 4)
    grade = _grade(score)

    notes = _quality_notes(
        unique_urls=unique_urls,
        online_sources=online_sources,
        offline_sources=offline_sources,
        unsupported_claims=unsupported_claims,
        total_claims=total_claims,
        score=score,
    )

    return EvidenceQualityReport(
        score=score,
        grade=grade,
        source_count=len(unique_urls),
        online_source_count=len(online_sources),
        offline_source_count=len(offline_sources),
        unsupported_claim_count=unsupported_claims,
        notes=notes,
    )


def _source_type_score(online_sources: list[str], offline_sources: list[str]) -> float:
    if not online_sources and not offline_sources:
        return 0.0

    source_scores: list[float] = []
    for url in online_sources:
        domain = _domain(url)
        if any(domain == known or domain.endswith(f".{known}") for known in _HIGH_QUALITY_DOMAINS):
            source_scores.append(1.0)
        elif any(domain == known or domain.endswith(f".{known}") for known in _MEDIUM_QUALITY_DOMAINS):
            source_scores.append(0.65)
        else:
            source_scores.append(0.75)

    for _ in offline_sources:
        source_scores.append(0.45)

    return sum(source_scores) / len(source_scores)


def _quality_notes(
    *,
    unique_urls: list[str],
    online_sources: list[str],
    offline_sources: list[str],
    unsupported_claims: int,
    total_claims: int,
    score: float,
) -> list[str]:
    notes: list[str] = []
    if not unique_urls:
        notes.append("No cited sources were returned by the evidence agents.")
    if offline_sources and not online_sources:
        notes.append("Evidence is based on local dataset similarity, not live verification.")
    if online_sources:
        notes.append(f"{len(online_sources)} online source(s) cited.")
    if unsupported_claims:
        notes.append(f"{unsupported_claims} of {total_claims} evidence item(s) are unresolved or uncited.")
    if score >= 0.75:
        notes.append("Evidence quality is strong enough to support a more decisive verdict.")
    elif score >= 0.45:
        notes.append("Evidence quality is moderate; verdict should still preserve uncertainty.")
    else:
        notes.append("Evidence quality is weak; treat the verdict as tentative.")
    return notes


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _grade(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.45:
        return "MEDIUM"
    return "LOW"
