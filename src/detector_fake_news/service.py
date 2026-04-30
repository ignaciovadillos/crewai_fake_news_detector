"""Application service for running the fake-news crew."""

from __future__ import annotations

import os
from contextlib import contextmanager
from hashlib import sha256
from collections.abc import Iterator
from typing import Any

from detector_fake_news.classifier import predict_baseline
from detector_fake_news.contradictions import detect_contradictions
from detector_fake_news.crew import build_crew
from detector_fake_news.evidence_quality import score_evidence_quality
from detector_fake_news.memory import build_memory_context, remember_analysis
from detector_fake_news.models import BiasReport, ClaimExtraction, FinalVerdict, LegalCase, PipelineReport

_REPORT_CACHE: dict[str, PipelineReport] = {}


def analyze_article(
    title: str,
    article_text: str,
    max_claims: int = 5,
    *,
    evidence_mode: str = "online",
    model_name: str | None = None,
    use_memory: bool = True,
    use_baseline: bool = True,
    use_cache: bool = True,
) -> PipelineReport:
    """Run the crew and convert task outputs into a typed pipeline report."""
    normalized_title = title.strip()
    normalized_text = article_text.strip()
    normalized_mode = _normalize_evidence_mode(evidence_mode)
    normalized_model = model_name.strip() if model_name else ""
    memory_context = (
        build_memory_context(
            title=normalized_title,
            article_text=normalized_text,
            evidence_mode=normalized_mode,
        )
        if use_memory
        else "Local analysis memory is disabled for this run."
    )
    cache_key = _cache_key(
        normalized_title,
        normalized_text,
        max_claims,
        normalized_mode,
        normalized_model,
        memory_context,
        str(use_baseline),
    )

    if use_cache and cache_key in _REPORT_CACHE:
        return _REPORT_CACHE[cache_key]

    with _temporary_evidence_mode(normalized_mode), _temporary_model(normalized_model):
        crew = build_crew()
        result = crew.kickoff(
            inputs={
                "title": normalized_title,
                "article_text": normalized_text,
                "max_claims": max_claims,
                "evidence_mode": normalized_mode,
                "model_name": normalized_model or "default",
                "memory_context": memory_context,
            }
        )
    (
        claims,
        supporting_case,
        opposing_case,
        bias_report,
        final_verdict,
    ) = _extract_structured_outputs(result.tasks_output)
    supporting_case, citation_note, downgraded_count = _enforce_supporting_citations(supporting_case)
    if downgraded_count:
        final_verdict = _apply_citation_penalty(
            final_verdict,
            downgraded_count=downgraded_count,
            note=citation_note,
        )
    contradiction_report = detect_contradictions(supporting_case, opposing_case)
    if contradiction_report.level in {"MEDIUM", "HIGH"}:
        final_verdict = _apply_contradiction_penalty(
            final_verdict,
            contradiction_report=contradiction_report,
        )
    final_verdict = _calibrate_final_verdict(
        final_verdict,
        supporting_case=supporting_case,
        opposing_case=opposing_case,
    )
    baseline_prediction = (
        predict_baseline(normalized_title, normalized_text)
        if use_baseline
        else None
    )
    evidence_quality = score_evidence_quality(supporting_case, opposing_case)

    report = PipelineReport(
        title=normalized_title,
        article_text=normalized_text,
        claims=claims,
        supporting_case=supporting_case,
        opposing_case=opposing_case,
        bias_report=bias_report,
        final_verdict=final_verdict,
        baseline_prediction=baseline_prediction,
        evidence_quality=evidence_quality,
        contradiction_report=contradiction_report,
        raw_output=result.raw,
    )
    if use_cache:
        _REPORT_CACHE[cache_key] = report
    if use_memory:
        remember_analysis(report, evidence_mode=normalized_mode)
    return report


def recommended_max_claims(*, fast_mode: bool) -> int:
    """Return a sensible claim budget for the selected execution mode."""
    return 3 if fast_mode else 5


def max_claims_for_depth(depth: str) -> int:
    """Return a claim budget for the requested analysis depth."""
    normalized = depth.strip().lower()
    if normalized == "quick":
        return 2
    if normalized == "deep":
        return 5
    return 3


def clear_report_cache() -> None:
    """Clear the in-memory analysis cache."""
    _REPORT_CACHE.clear()


def _coerce_model(value: Any, model_type: type[Any]) -> Any:
    if isinstance(value, model_type):
        return value
    if value is None:
        raise ValueError(
            f"Expected CrewAI to return {model_type.__name__} but received no structured output."
        )
    if hasattr(value, "model_dump"):
        return model_type.model_validate(value.model_dump())
    return model_type.model_validate(value)


def _extract_structured_outputs(
    task_outputs: list[Any],
) -> tuple[ClaimExtraction, LegalCase, LegalCase, BiasReport, FinalVerdict]:
    """Map CrewAI task outputs by schema instead of list position.

    Async CrewAI tasks can complete out of order, so relying on positional task
    output indexes can accidentally parse a supporting case as claim extraction.
    """
    claims: ClaimExtraction | None = None
    supporting_case: LegalCase | None = None
    opposing_case: LegalCase | None = None
    bias_report: BiasReport | None = None
    final_verdict: FinalVerdict | None = None

    for task_output in task_outputs:
        value = getattr(task_output, "pydantic", task_output)
        if value is None:
            continue

        parsed_claims = _try_coerce_model(value, ClaimExtraction)
        if parsed_claims is not None:
            claims = parsed_claims
            continue

        parsed_case = _try_coerce_model(value, LegalCase)
        if parsed_case is not None:
            if parsed_case.stance == "supporting":
                supporting_case = parsed_case
            elif parsed_case.stance == "opposing":
                opposing_case = parsed_case
            continue

        parsed_bias = _try_coerce_model(value, BiasReport)
        if parsed_bias is not None:
            bias_report = parsed_bias
            continue

        parsed_verdict = _try_coerce_model(value, FinalVerdict)
        if parsed_verdict is not None:
            final_verdict = parsed_verdict

    if claims is None:
        recovered_claims = _recover_claims_from_cases(supporting_case, opposing_case)
        if recovered_claims:
            claims = ClaimExtraction(claims=recovered_claims[:5])

    missing = []
    if claims is None:
        missing.append("ClaimExtraction")
    if supporting_case is None:
        missing.append("supporting LegalCase")
    if opposing_case is None:
        missing.append("opposing LegalCase")
    if bias_report is None:
        missing.append("BiasReport")
    if final_verdict is None:
        missing.append("FinalVerdict")
    if missing:
        raise ValueError(
            "CrewAI did not return all expected structured outputs: "
            + ", ".join(missing)
        )

    return claims, supporting_case, opposing_case, bias_report, final_verdict


def _try_coerce_model(value: Any, model_type: type[Any]) -> Any | None:
    try:
        return _coerce_model(value, model_type)
    except Exception:
        return None


def _recover_claims_from_cases(
    supporting_case: LegalCase | None,
    opposing_case: LegalCase | None,
) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for legal_case in (supporting_case, opposing_case):
        if legal_case is None:
            continue
        for item in legal_case.results:
            normalized = item.claim.strip()
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            claims.append(normalized)
            seen.add(key)
    return claims


def _cache_key(
    title: str,
    article_text: str,
    max_claims: int,
    evidence_mode: str,
    model_name: str,
    memory_context: str,
    use_baseline: str,
) -> str:
    payload = "\n".join(
        [
            title,
            article_text,
            str(max_claims),
            evidence_mode,
            model_name,
            memory_context,
            use_baseline,
        ]
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _normalize_evidence_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"offline", "hybrid"}:
        return normalized
    return "online"


@contextmanager
def _temporary_evidence_mode(mode: str) -> Iterator[None]:
    previous_value = os.environ.get("DETECTOR_EVIDENCE_MODE")
    os.environ["DETECTOR_EVIDENCE_MODE"] = mode
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("DETECTOR_EVIDENCE_MODE", None)
        else:
            os.environ["DETECTOR_EVIDENCE_MODE"] = previous_value


@contextmanager
def _temporary_model(model_name: str) -> Iterator[None]:
    previous_value = os.environ.get("MODEL")
    if model_name:
        os.environ["MODEL"] = model_name
    try:
        yield
    finally:
        if model_name:
            if previous_value is None:
                os.environ.pop("MODEL", None)
            else:
                os.environ["MODEL"] = previous_value


def _calibrate_final_verdict(
    verdict: FinalVerdict,
    *,
    supporting_case: LegalCase,
    opposing_case: LegalCase,
) -> FinalVerdict:
    """Align the categorical label with the truth score and claim evidence."""
    contradiction_count = sum(1 for item in opposing_case.results if item.verdict == "CONTRADICTED")
    support_count = sum(1 for item in supporting_case.results if item.verdict == "SUPPORTED")
    has_meaningful_split = contradiction_count > 0 and support_count > 0

    adjusted_label = verdict.label
    calibration_note = ""

    if verdict.label == "MIXED" and verdict.truth_score >= 0.75 and not has_meaningful_split:
        adjusted_label = "REAL"
        calibration_note = (
            "Label auto-calibrated from MIXED to REAL because the truth score was high and "
            "the claim evidence did not show a strong split."
        )
    elif verdict.label == "MIXED" and verdict.truth_score <= 0.25 and not has_meaningful_split:
        adjusted_label = "FAKE"
        calibration_note = (
            "Label auto-calibrated from MIXED to FAKE because the truth score was low and "
            "the claim evidence did not show a strong split."
        )
    elif verdict.label == "REAL" and verdict.truth_score < 0.6:
        adjusted_label = "MIXED"
        calibration_note = (
            "Label auto-calibrated from REAL to MIXED because the truth score was not high "
            "enough for a decisive positive verdict."
        )
    elif verdict.label == "FAKE" and verdict.truth_score > 0.4:
        adjusted_label = "MIXED"
        calibration_note = (
            "Label auto-calibrated from FAKE to MIXED because the truth score was not low "
            "enough for a decisive negative verdict."
        )
    elif verdict.label == "UNVERIFIABLE" and (support_count > 0 or contradiction_count > 0):
        if verdict.truth_score >= 0.6:
            adjusted_label = "REAL"
            calibration_note = (
                "Label auto-calibrated from UNVERIFIABLE to REAL because external evidence "
                "was present and the truth score was strong."
            )
        elif verdict.truth_score <= 0.4:
            adjusted_label = "FAKE"
            calibration_note = (
                "Label auto-calibrated from UNVERIFIABLE to FAKE because external evidence "
                "was present and the truth score was strongly negative."
            )
        else:
            adjusted_label = "MIXED"
            calibration_note = (
                "Label auto-calibrated from UNVERIFIABLE to MIXED because external evidence "
                "was present but the overall score remained uncertain."
            )

    if adjusted_label == verdict.label:
        return verdict

    explanation = verdict.explanation
    summary = verdict.summary
    if calibration_note not in explanation:
        explanation = f"{explanation}\n\n{calibration_note}"
    if calibration_note not in summary:
        summary = f"{summary} ({calibration_note})"

    return verdict.model_copy(
        update={
            "label": adjusted_label,
            "summary": summary,
            "explanation": explanation,
        }
    )


def _enforce_supporting_citations(supporting_case: LegalCase) -> tuple[LegalCase, str, int]:
    """Downgrade supported evidence items that do not cite an outside source."""
    changed_results = []
    downgraded_count = 0

    for item in supporting_case.results:
        if item.verdict == "SUPPORTED" and not item.source_urls:
            downgraded_count += 1
            changed_results.append(
                item.model_copy(
                    update={
                        "verdict": "UNVERIFIABLE",
                        "confidence": min(item.confidence, 0.5),
                        "reasoning": (
                            f"{item.reasoning} Citation enforcement: this was downgraded "
                            "from SUPPORTED to UNVERIFIABLE because no source_urls were provided."
                        ),
                    }
                )
            )
        else:
            changed_results.append(item)

    if downgraded_count == 0:
        return supporting_case, "", 0

    note = (
        f"Citation enforcement downgraded {downgraded_count} supporting claim(s) "
        "because they were marked SUPPORTED without source URLs."
    )
    return (
        supporting_case.model_copy(
            update={
                "case_summary": f"{supporting_case.case_summary} {note}",
                "results": changed_results,
            }
        ),
        note,
        downgraded_count,
    )


def _apply_citation_penalty(
    verdict: FinalVerdict,
    *,
    downgraded_count: int,
    note: str,
) -> FinalVerdict:
    truth_penalty = min(0.10 * downgraded_count, 0.30)
    confidence_penalty = min(0.05 * downgraded_count, 0.20)
    explanation = verdict.explanation
    summary = verdict.summary
    if note not in explanation:
        explanation = f"{explanation}\n\n{note}"
    if note not in summary:
        summary = f"{summary} ({note})"
    return verdict.model_copy(
        update={
            "truth_score": max(verdict.truth_score - truth_penalty, 0.0),
            "confidence": max(verdict.confidence - confidence_penalty, 0.0),
            "summary": summary,
            "explanation": explanation,
        }
    )


def _apply_contradiction_penalty(
    verdict: FinalVerdict,
    *,
    contradiction_report,
) -> FinalVerdict:
    penalty = 0.12 if contradiction_report.level == "MEDIUM" else 0.22
    note = (
        f"Contradiction detector found {contradiction_report.level.lower()} conflict "
        f"across evidence cases: {contradiction_report.contradiction_count} strong "
        f"contradiction(s), {contradiction_report.mixed_signal_count} mixed signal(s)."
    )
    adjusted_label = verdict.label
    adjusted_truth = max(verdict.truth_score - penalty, 0.0)
    if contradiction_report.level == "HIGH" and verdict.label == "REAL":
        adjusted_label = "MIXED"

    explanation = verdict.explanation
    summary = verdict.summary
    if note not in explanation:
        explanation = f"{explanation}\n\n{note}"
    if note not in summary:
        summary = f"{summary} ({note})"

    return verdict.model_copy(
        update={
            "label": adjusted_label,
            "truth_score": adjusted_truth,
            "confidence": max(verdict.confidence - 0.05, 0.0),
            "summary": summary,
            "explanation": explanation,
        }
    )
