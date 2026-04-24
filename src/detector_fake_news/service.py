"""Application service for running the fake-news crew."""

from __future__ import annotations

from typing import Any

from detector_fake_news.crew import build_crew
from detector_fake_news.models import BiasReport, ClaimExtraction, FinalVerdict, LegalCase, PipelineReport


def analyze_article(title: str, article_text: str, max_claims: int = 5) -> PipelineReport:
    """Run the crew and convert task outputs into a typed pipeline report."""
    crew = build_crew()
    result = crew.kickoff(
        inputs={
            "title": title.strip(),
            "article_text": article_text.strip(),
            "max_claims": max_claims,
        }
    )
    task_outputs = result.tasks_output

    claims = _coerce_model(task_outputs[0].pydantic, ClaimExtraction)
    supporting_case = _coerce_model(task_outputs[1].pydantic, LegalCase)
    opposing_case = _coerce_model(task_outputs[2].pydantic, LegalCase)
    bias_report = _coerce_model(task_outputs[3].pydantic, BiasReport)
    final_verdict = _coerce_model(task_outputs[4].pydantic, FinalVerdict)

    return PipelineReport(
        title=title.strip(),
        article_text=article_text.strip(),
        claims=claims,
        supporting_case=supporting_case,
        opposing_case=opposing_case,
        bias_report=bias_report,
        final_verdict=final_verdict,
        raw_output=result.raw,
    )


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
