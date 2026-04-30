"""Report export helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from detector_fake_news.models import PipelineReport


def report_to_markdown(report: PipelineReport) -> str:
    """Render a complete article analysis as Markdown."""
    verdict = report.final_verdict
    lines = [
        "# Fake News Analysis Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Title: {report.title or 'Untitled article'}",
        "",
        "## Final Verdict",
        "",
        f"- Label: {verdict.label}",
        f"- Truth score: {verdict.truth_score:.0%}",
        f"- Confidence: {verdict.confidence:.0%}",
        f"- Bias score: {verdict.bias_score:.0%}",
        f"- Tone: {verdict.tone}",
        "",
        verdict.summary,
        "",
        "### Explanation",
        "",
        verdict.explanation,
        "",
    ]

    if report.evidence_quality:
        quality = report.evidence_quality
        lines.extend(
            [
                "## Evidence Quality",
                "",
                f"- Grade: {quality.grade}",
                f"- Score: {quality.score:.0%}",
                f"- Total sources: {quality.source_count}",
                f"- Online sources: {quality.online_source_count}",
                f"- Offline sources: {quality.offline_source_count}",
                f"- Unresolved or uncited evidence items: {quality.unsupported_claim_count}",
                "",
                *[f"- {note}" for note in quality.notes],
                "",
            ]
        )

    if report.contradiction_report:
        contradiction = report.contradiction_report
        lines.extend(
            [
                "## Contradiction Analysis",
                "",
                f"- Level: {contradiction.level}",
                f"- Score: {contradiction.score:.0%}",
                f"- Strong contradictions: {contradiction.contradiction_count}",
                f"- Mixed signals: {contradiction.mixed_signal_count}",
                "",
                *[f"- {note}" for note in contradiction.notes],
                "",
            ]
        )

    if report.baseline_prediction:
        baseline = report.baseline_prediction
        lines.extend(
            [
                "## Classifier Baseline",
                "",
                f"- Label: {baseline.label}",
                f"- Confidence: {baseline.confidence:.0%}",
                f"- Fake probability: {baseline.fake_probability:.0%}",
                f"- Real probability: {baseline.real_probability:.0%}",
                f"- Training examples: {baseline.training_examples}",
                f"- Top indicators: {', '.join(baseline.top_indicators) or 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Extracted Claims",
            "",
            *[f"{index}. {claim}" for index, claim in enumerate(report.claims.claims, start=1)],
            "",
            "## Supporting Case",
            "",
            report.supporting_case.case_summary,
            "",
            _legal_case_table(report.supporting_case),
            "",
            "## Opposing Case",
            "",
            report.opposing_case.case_summary,
            "",
            _legal_case_table(report.opposing_case),
            "",
            "## Bias Analysis",
            "",
            f"- Tone: {report.bias_report.tone}",
            f"- Bias score: {report.bias_report.bias_score:.0%}",
            f"- Flags: {', '.join(report.bias_report.flags) or 'None'}",
            "",
            report.bias_report.reasoning,
            "",
            "## Article Text",
            "",
            report.article_text,
            "",
        ]
    )
    return "\n".join(lines)


def report_filename(title: str) -> str:
    safe_title = "".join(
        char.lower() if char.isalnum() else "-"
        for char in title.strip()
    ).strip("-")
    safe_title = "-".join(part for part in safe_title.split("-") if part)
    if not safe_title:
        safe_title = "article-analysis"
    return f"{safe_title[:80]}-report.md"


def _legal_case_table(legal_case: object) -> str:
    rows = [
        "| # | Claim | Verdict | Confidence | Sources | Evidence |",
        "|---|---|---|---:|---|---|",
    ]
    for index, item in enumerate(getattr(legal_case, "results", []), start=1):
        sources = "<br>".join(item.source_urls) if item.source_urls else "No source URLs"
        rows.append(
            "| "
            f"{index} | "
            f"{_escape_table(item.claim)} | "
            f"{item.verdict} | "
            f"{item.confidence:.0%} | "
            f"{_escape_table(sources)} | "
            f"{_escape_table(item.evidence)} |"
        )
    return "\n".join(rows)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
