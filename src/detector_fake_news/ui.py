"""Streamlit UI for the fake-news detector."""

from __future__ import annotations

import csv
import io
import json
import time

import streamlit as st

from detector_fake_news.article_fetcher import fetch_article_from_url
from detector_fake_news.diagnostics import (
    append_batch_row_log,
    batch_log_filename,
    batch_log_path,
    describe_exception,
    new_batch_run_id,
    utc_timestamp,
)
from detector_fake_news.history import clear_run_history, recent_runs, record_batch_run, record_single_run
from detector_fake_news.memory import clear_analysis_memory
from detector_fake_news.reporting import report_filename, report_to_markdown
from detector_fake_news.service import analyze_article, max_claims_for_depth
from detector_fake_news.ui_helpers import (
    baseline_agreement as _baseline_agreement,
    expected_match as _expected_match,
    normalize_expected_label as _normalize_expected_label,
    rows_to_csv as _rows_to_csv,
)

_MODEL_OPTIONS = [
    "Default from .env",
    "ollama/qwen2.5:7b",
    "ollama/llama3.1:8b",
    "ollama/mistral:7b",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "Custom",
]


st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
)


def main() -> None:
    st.title("Multi-Agent Fake News Detector")
    st.caption(
        "A simple CrewAI MVP with claim extraction, supporting and opposing cases, "
        "bias analysis, and a final judge."
    )

    with st.sidebar:
        st.subheader("Workflow")
        st.markdown(
            """
            1. Extract the main factual claims
            2. Build the case for credibility
            3. Build the case for skepticism
            4. Analyze tone and bias
            5. Ask the judge for a final verdict
            """
        )
        st.info(
            "Claim extraction runs first, then the supporting and opposing case "
            "tasks run in parallel before the final verdict."
        )
        if st.button("Clear local memory", use_container_width=True):
            clear_analysis_memory()
            st.success("Local analysis memory cleared.")
        render_run_history_sidebar()

    single_tab, batch_tab = st.tabs(["Single Article", "Batch CSV"])

    with single_tab:
        render_single_article()

    with batch_tab:
        render_batch_mode()


def render_single_article() -> None:
    evidence_mode, use_memory, use_baseline, analysis_depth, model_name = _render_single_settings()
    title = st.text_input("Article title", placeholder="Enter the news headline")
    article_url = st.text_input("Article URL", placeholder="Optional: paste a news article URL")
    article_text = st.text_area(
        "Article text",
        height=280,
        placeholder="Paste the article body here...",
    )

    uploaded_txt = st.file_uploader(
        "Or upload a .txt or .md file",
        type=["txt", "md"],
        accept_multiple_files=False,
    )
    if uploaded_txt is not None:
        article_text = uploaded_txt.getvalue().decode("utf-8")
        st.success("Loaded article text from uploaded file.")

    if st.button("Analyze article", type="primary", use_container_width=True):
        title, article_text = _resolve_article_input(
            title=title,
            article_text=article_text,
            article_url=article_url,
        )
        if not article_text.strip():
            st.error("Please provide article text or a URL before running the analysis.")
            return

        with st.spinner("Running the agentic workflow..."):
            report = analyze_article(
                title=title,
                article_text=article_text,
                max_claims=max_claims_for_depth(analysis_depth),
                evidence_mode=evidence_mode,
                model_name=model_name,
                use_memory=use_memory,
                use_baseline=use_baseline,
            )
            record_single_run(
                report,
                evidence_mode=evidence_mode,
                model_name=model_name,
                analysis_depth=analysis_depth,
                use_memory=use_memory,
            )

        st.subheader("Final Verdict")
        verdict = report.final_verdict
        truth_score_pct = verdict.truth_score * 100

        st.markdown(
            f"### Truth Score: `{truth_score_pct:.1f}%`"
        )
        st.progress(verdict.truth_score)
        st.caption(_truth_score_caption(verdict.truth_score))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Label", verdict.label)
        col2.metric("Truth score", f"{truth_score_pct:.1f}%")
        col3.metric("Confidence", f"{verdict.confidence:.0%}")
        col4.metric("Bias score", f"{verdict.bias_score:.0%}")
        st.write(verdict.summary)

        if report.evidence_quality is not None:
            quality = report.evidence_quality
            st.subheader("Evidence Quality")
            quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
            quality_col1.metric("Grade", quality.grade)
            quality_col2.metric("Score", f"{quality.score:.0%}")
            quality_col3.metric("Sources", quality.source_count)
            quality_col4.metric("Unresolved/uncited", quality.unsupported_claim_count)
            if quality.notes:
                st.write(" ".join(quality.notes))

        if report.contradiction_report is not None:
            contradiction = report.contradiction_report
            st.subheader("Contradiction Analysis")
            conflict_col1, conflict_col2, conflict_col3, conflict_col4 = st.columns(4)
            conflict_col1.metric("Level", contradiction.level)
            conflict_col2.metric("Score", f"{contradiction.score:.0%}")
            conflict_col3.metric("Strong conflicts", contradiction.contradiction_count)
            conflict_col4.metric("Mixed signals", contradiction.mixed_signal_count)
            if contradiction.notes:
                st.write(" ".join(contradiction.notes))

        if report.baseline_prediction is not None:
            baseline = report.baseline_prediction
            st.subheader("Classifier Baseline")
            base_col1, base_col2, base_col3, base_col4 = st.columns(4)
            base_col1.metric("Baseline label", baseline.label)
            base_col2.metric("Fake probability", f"{baseline.fake_probability:.0%}")
            base_col3.metric("Real probability", f"{baseline.real_probability:.0%}")
            base_col4.metric("Agreement", _baseline_agreement(verdict.label, baseline.label))
            st.caption(
                "Baseline is a local bag-of-words model trained from the Kaggle archive; "
                "it is a comparison signal, not a substitute for fact-checking."
            )
            if baseline.top_indicators:
                st.write("Top baseline indicators:", ", ".join(baseline.top_indicators))

        report_markdown = report_to_markdown(report)
        st.download_button(
            "Download full Markdown report",
            data=report_markdown,
            file_name=report_filename(report.title),
            mime="text/markdown",
            use_container_width=True,
        )

        timeline_tab, judge_tab, claims_tab, support_tab, oppose_tab, bias_tab = st.tabs(
            ["Timeline", "Judge", "Claims", "Supporting Case", "Opposing Case", "Bias"]
        )

        with timeline_tab:
            st.dataframe(_analysis_timeline_rows(report), use_container_width=True)

        with judge_tab:
            st.json(verdict.model_dump())

        with claims_tab:
            st.json(report.claims.model_dump())

        with support_tab:
            st.write(report.supporting_case.case_summary)
            st.dataframe(_evidence_table_rows(report.supporting_case), use_container_width=True)
            with st.expander("Raw supporting JSON"):
                st.json(report.supporting_case.model_dump())

        with oppose_tab:
            st.write(report.opposing_case.case_summary)
            st.dataframe(_evidence_table_rows(report.opposing_case), use_container_width=True)
            with st.expander("Raw opposing JSON"):
                st.json(report.opposing_case.model_dump())

        with bias_tab:
            st.json(report.bias_report.model_dump())

    st.divider()
    if st.button("Compare evidence modes", use_container_width=True):
        title, article_text = _resolve_article_input(
            title=title,
            article_text=article_text,
            article_url=article_url,
        )
        if not article_text.strip():
            st.error("Please provide article text or a URL before running the comparison.")
            return

        comparison_rows: list[dict[str, str]] = []
        progress = st.progress(0.0)
        status = st.empty()
        for index, mode in enumerate(["offline", "online", "hybrid"], start=1):
            status.write(f"Running {mode} mode...")
            try:
                report = analyze_article(
                    title=title,
                    article_text=article_text,
                    max_claims=max_claims_for_depth(analysis_depth),
                    evidence_mode=mode,
                    model_name=model_name,
                    use_memory=use_memory,
                    use_baseline=use_baseline,
                )
                comparison_rows.append(
                    _comparison_row(
                        mode,
                        report,
                        analysis_depth=analysis_depth,
                        model_name=model_name,
                    )
                )
            except Exception as exc:
                comparison_rows.append(
                    {
                        "mode": mode,
                        "status": "ERROR",
                        "label": "ERROR",
                        "truth_score": "",
                        "confidence": "",
                        "bias_score": "",
                        "baseline_label": "",
                        "baseline_agreement": "",
                        "supporting_sources": "",
                        "opposing_sources": "",
                        "summary": str(exc),
                    }
                )
            progress.progress(index / 3)

        status.empty()
        st.subheader("Evidence Mode Comparison")
        st.dataframe(comparison_rows, use_container_width=True)
        st.download_button(
            "Download comparison CSV",
            data=_rows_to_csv(comparison_rows),
            file_name="mode_comparison.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _render_single_settings() -> tuple[str, bool, bool, str, str]:
    with st.container(border=True):
        st.caption("Analysis settings")
        mode_col, depth_col, model_col, memory_col, baseline_col = st.columns(
            [1.1, 1.1, 1.6, 0.9, 1.1],
            vertical_alignment="bottom",
        )
        with mode_col:
            evidence_mode = st.selectbox(
                "Evidence mode",
                options=["online", "offline", "hybrid"],
                index=0,
                help=(
                    "Online uses web evidence. Offline searches the local Kaggle archive. "
                    "Hybrid combines both."
                ),
                key="single_evidence_mode",
            )
        with depth_col:
            analysis_depth = st.selectbox(
                "Analysis depth",
                options=["Quick", "Standard", "Deep"],
                index=1,
                help="Quick extracts fewer claims. Deep checks more claims and usually takes longer.",
                key="single_analysis_depth",
            )
        with model_col:
            model_name = _model_selector("single")
        with memory_col:
            use_memory = st.checkbox(
                "Memory",
                value=True,
                help="Looks up similar articles analyzed earlier and stores this result for future runs.",
                key="single_use_memory",
            )
        with baseline_col:
            use_baseline = st.checkbox(
                "Baseline",
                value=True,
                help="Compares the agent verdict with a local Naive Bayes model trained from the Kaggle archive.",
                key="single_use_baseline",
            )
    return evidence_mode, use_memory, use_baseline, analysis_depth, model_name


def render_batch_mode() -> None:
    st.write(
        "Upload a CSV with `article_text`/`text` or `url`/`article_url`, and optionally `title`."
    )
    (
        evidence_mode,
        analysis_depth,
        model_name,
        use_memory,
        use_baseline,
        include_detailed_outputs,
        slow_row_threshold,
    ) = _render_batch_settings()
    uploaded_csv = st.file_uploader(
        "Upload batch CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="batch_uploader",
    )

    if uploaded_csv is None:
        return

    rows = list(csv.DictReader(io.StringIO(uploaded_csv.getvalue().decode("utf-8"))))
    st.write(f"Loaded {len(rows)} rows.")

    if st.button("Run batch analysis", use_container_width=True):
        outputs: list[dict[str, str]] = []
        diagnostic_rows: list[dict[str, str]] = []
        progress = st.progress(0.0)
        status = st.empty()
        max_claims = max_claims_for_depth(analysis_depth)
        batch_run_id = new_batch_run_id()

        for index, row in enumerate(rows, start=1):
            row_started_at = utc_timestamp()
            row_start_perf = time.perf_counter()
            analysis_start_perf: float | None = None
            phase = "reading input"
            status.write(f"Processing row {index} of {len(rows)}...")
            title = (row.get("title") or "").strip()
            article_text = (row.get("article_text") or row.get("text") or "").strip()
            article_url = (row.get("article_url") or row.get("url") or "").strip()
            expected_label = (
                row.get("expected_label")
                or row.get("ground_truth")
                or row.get("label")
                or ""
            ).strip()
            if not article_text and article_url:
                phase = "fetching url"
                try:
                    fetched = fetch_article_from_url(article_url)
                    article_text = fetched.text
                    if not title:
                        title = fetched.title
                except Exception as exc:
                    output_row = {
                        "title": title,
                        "url": article_url,
                        "expected_label": expected_label,
                        "evidence_mode": evidence_mode,
                        "model_name": model_name or "default",
                        "analysis_depth": analysis_depth,
                        "memory_enabled": str(use_memory),
                        "baseline_enabled": str(use_baseline),
                        "status": "ERROR",
                        "label": "ERROR",
                        "expected_match": _expected_match(expected_label, "ERROR"),
                        "truth_score": "",
                        "confidence": "",
                        "summary": f"URL fetch failed: {exc}",
                        "error": str(exc),
                    }
                    outputs.append(
                        _finalize_batch_output(
                            output_row,
                            diagnostic_rows,
                            batch_run_id=batch_run_id,
                            row_index=index,
                            total_rows=len(rows),
                            started_at=row_started_at,
                            start_perf=row_start_perf,
                            phase=phase,
                            slow_threshold_seconds=float(slow_row_threshold),
                            article_text=article_text,
                            error=exc,
                        )
                    )
                    progress.progress(index / max(len(rows), 1))
                    continue

            if not article_text:
                phase = "validating input"
                output_row = {
                    "title": title,
                    "url": article_url,
                    "expected_label": expected_label,
                    "evidence_mode": evidence_mode,
                    "model_name": model_name or "default",
                    "analysis_depth": analysis_depth,
                    "memory_enabled": str(use_memory),
                    "baseline_enabled": str(use_baseline),
                    "status": "SKIPPED",
                    "label": "SKIPPED",
                    "expected_match": _expected_match(expected_label, "SKIPPED"),
                    "truth_score": "0.0",
                    "confidence": "0.0",
                    "baseline_label": "",
                    "baseline_fake_probability": "",
                    "baseline_real_probability": "",
                    "baseline_confidence": "",
                    "baseline_agreement": "",
                    "evidence_quality_score": "",
                    "evidence_quality_grade": "",
                    "evidence_source_count": "",
                    "contradiction_score": "",
                    "contradiction_level": "",
                    "summary": "Missing article_text/text column value.",
                }
                outputs.append(
                    _finalize_batch_output(
                        output_row,
                        diagnostic_rows,
                        batch_run_id=batch_run_id,
                        row_index=index,
                        total_rows=len(rows),
                        started_at=row_started_at,
                        start_perf=row_start_perf,
                        phase=phase,
                        slow_threshold_seconds=float(slow_row_threshold),
                        article_text=article_text,
                    )
                )
                progress.progress(index / max(len(rows), 1))
                continue

            try:
                phase = "running analysis"
                analysis_start_perf = time.perf_counter()
                report = analyze_article(
                    title=title,
                    article_text=article_text,
                    max_claims=max_claims,
                    evidence_mode=evidence_mode,
                    model_name=model_name,
                    use_memory=use_memory,
                    use_baseline=use_baseline,
                )
                analysis_duration_seconds = time.perf_counter() - analysis_start_perf
                phase = "formatting output"
                baseline = report.baseline_prediction
                output_row = {
                    "title": title,
                    "url": article_url,
                    "expected_label": expected_label,
                    "evidence_mode": evidence_mode,
                    "model_name": model_name or "default",
                    "analysis_depth": analysis_depth,
                    "memory_enabled": str(use_memory),
                    "baseline_enabled": str(use_baseline),
                    "analysis_duration_seconds": f"{analysis_duration_seconds:.2f}",
                    "status": "OK",
                    "label": report.final_verdict.label,
                    "expected_match": _expected_match(expected_label, report.final_verdict.label),
                    "truth_score": f"{report.final_verdict.truth_score:.4f}",
                    "confidence": f"{report.final_verdict.confidence:.4f}",
                    "baseline_label": baseline.label if baseline else "",
                    "baseline_fake_probability": (
                        f"{baseline.fake_probability:.4f}" if baseline else ""
                    ),
                    "baseline_real_probability": (
                        f"{baseline.real_probability:.4f}" if baseline else ""
                    ),
                    "baseline_confidence": f"{baseline.confidence:.4f}" if baseline else "",
                    "baseline_agreement": (
                        _baseline_agreement(report.final_verdict.label, baseline.label)
                        if baseline
                        else ""
                    ),
                    "evidence_quality_score": (
                        f"{report.evidence_quality.score:.4f}"
                        if report.evidence_quality
                        else ""
                    ),
                    "evidence_quality_grade": (
                        report.evidence_quality.grade
                        if report.evidence_quality
                        else ""
                    ),
                    "evidence_source_count": (
                        str(report.evidence_quality.source_count)
                        if report.evidence_quality
                        else ""
                    ),
                    "contradiction_score": (
                        f"{report.contradiction_report.score:.4f}"
                        if report.contradiction_report
                        else ""
                    ),
                    "contradiction_level": (
                        report.contradiction_report.level
                        if report.contradiction_report
                        else ""
                    ),
                    "bias_score": f"{report.bias_report.bias_score:.4f}",
                    "tone": report.bias_report.tone,
                    "summary": report.final_verdict.summary,
                    "claims": json.dumps(report.claims.model_dump(), ensure_ascii=True),
                    "supporting_source_count": str(_source_count(report.supporting_case)),
                    "opposing_source_count": str(_source_count(report.opposing_case)),
                }
                if include_detailed_outputs:
                    output_row.update(
                        {
                            "explanation": report.final_verdict.explanation,
                            "supporting_case": json.dumps(
                                report.supporting_case.model_dump(),
                                ensure_ascii=True,
                            ),
                            "opposing_case": json.dumps(
                                report.opposing_case.model_dump(),
                                ensure_ascii=True,
                            ),
                            "bias_report": json.dumps(
                                report.bias_report.model_dump(),
                                ensure_ascii=True,
                            ),
                        }
                    )
                outputs.append(
                    _finalize_batch_output(
                        output_row,
                        diagnostic_rows,
                        batch_run_id=batch_run_id,
                        row_index=index,
                        total_rows=len(rows),
                        started_at=row_started_at,
                        start_perf=row_start_perf,
                        phase=phase,
                        slow_threshold_seconds=float(slow_row_threshold),
                        article_text=article_text,
                    )
                )
            except Exception as exc:
                analysis_duration_seconds = (
                    time.perf_counter() - analysis_start_perf
                    if analysis_start_perf is not None
                    else 0.0
                )
                output_row = {
                        "title": title,
                        "url": article_url,
                        "expected_label": expected_label,
                        "evidence_mode": evidence_mode,
                        "model_name": model_name or "default",
                        "analysis_depth": analysis_depth,
                        "memory_enabled": str(use_memory),
                        "baseline_enabled": str(use_baseline),
                        "analysis_duration_seconds": f"{analysis_duration_seconds:.2f}",
                        "status": "ERROR",
                        "label": "ERROR",
                        "expected_match": _expected_match(expected_label, "ERROR"),
                        "truth_score": "",
                        "confidence": "",
                        "bias_score": "",
                        "tone": "",
                        "summary": "Batch row failed during analysis.",
                        "claims": "",
                        "supporting_source_count": "",
                        "opposing_source_count": "",
                        "baseline_label": "",
                        "baseline_fake_probability": "",
                        "baseline_real_probability": "",
                        "baseline_confidence": "",
                        "baseline_agreement": "",
                        "evidence_quality_score": "",
                        "evidence_quality_grade": "",
                        "evidence_source_count": "",
                        "contradiction_score": "",
                        "contradiction_level": "",
                        "error": str(exc),
                    }
                outputs.append(
                    _finalize_batch_output(
                        output_row,
                        diagnostic_rows,
                        batch_run_id=batch_run_id,
                        row_index=index,
                        total_rows=len(rows),
                        started_at=row_started_at,
                        start_perf=row_start_perf,
                        phase=phase,
                        slow_threshold_seconds=float(slow_row_threshold),
                        article_text=article_text,
                        error=exc,
                    )
                )
            progress.progress(index / max(len(rows), 1))

        status.empty()
        st.success("Batch run finished.")
        record_batch_run(
            outputs,
            evidence_mode=evidence_mode,
            model_name=model_name,
            analysis_depth=analysis_depth,
            use_memory=use_memory,
        )
        _render_batch_metrics(outputs)
        _render_batch_diagnostics(outputs, diagnostic_rows, batch_run_id)
        filtered_outputs = _filter_batch_outputs(outputs)
        _render_batch_results_table(outputs, filtered_outputs)


def _render_batch_settings() -> tuple[str, str, str, bool, bool, bool, int]:
    with st.container(border=True):
        st.caption("Batch settings")
        mode_col, depth_col, model_col, threshold_col, memory_col, baseline_col, detail_col = st.columns(
            [1.0, 1.0, 1.45, 1.0, 0.75, 0.85, 0.85],
            vertical_alignment="bottom",
        )
        with mode_col:
            evidence_mode = st.selectbox(
                "Evidence mode",
                options=["online", "offline", "hybrid"],
                index=1,
                help=(
                    "Online uses web evidence. Offline searches the local Kaggle archive. "
                    "Hybrid combines both."
                ),
                key="batch_evidence_mode",
            )
        with depth_col:
            analysis_depth = st.selectbox(
                "Analysis depth",
                options=["Quick", "Standard", "Deep"],
                index=0,
                help="Quick is best for Ollama batch runs. Deep checks more claims and takes longer.",
                key="batch_analysis_depth",
            )
        with model_col:
            model_name = _model_selector("batch")
        with threshold_col:
            slow_row_threshold = st.number_input(
                "Slow row (s)",
                min_value=10,
                max_value=900,
                value=120,
                step=10,
                help="Rows taking longer than this are flagged in diagnostics and the results CSV.",
                key="batch_slow_row_threshold",
            )
        with memory_col:
            use_memory = st.checkbox(
                "Memory",
                value=False,
                help="Uses previous analyses as hints. Leave off for cleaner dataset evaluation.",
                key="batch_use_memory",
            )
        with baseline_col:
            use_baseline = st.checkbox(
                "Baseline",
                value=True,
                help="Adds a local Naive Bayes prediction trained from the Kaggle archive.",
                key="batch_use_baseline",
            )
        with detail_col:
            include_detailed_outputs = st.checkbox(
                "Details",
                value=True,
                help="Adds supporting, opposing, bias, and explanation fields to the output table and CSV.",
                key="batch_include_detailed_outputs",
            )
    return (
        evidence_mode,
        analysis_depth,
        model_name,
        use_memory,
        use_baseline,
        include_detailed_outputs,
        slow_row_threshold,
    )


def render_run_history_sidebar() -> None:
    st.subheader("Run History")
    runs = recent_runs(limit=6)
    if not runs:
        st.caption("No saved runs yet.")
    for run in runs:
        label = run.get("label") or f"{run.get('completed', 0)}/{run.get('rows', 0)} completed"
        title = run.get("title") or "Untitled run"
        created_at = str(run.get("created_at", ""))[:19].replace("T", " ")
        with st.expander(f"{label} - {title[:42]}"):
            st.caption(created_at)
            st.write(f"Mode: {run.get('evidence_mode', 'unknown')}")
            st.write(f"Model: {run.get('model_name', 'default')}")
            st.write(f"Depth: {run.get('analysis_depth', 'unknown')}")
            if run.get("run_type") == "single":
                st.write(f"Truth: {float(run.get('truth_score', 0.0)):.0%}")
                st.write(f"Confidence: {float(run.get('confidence', 0.0)):.0%}")
                st.write(run.get("summary", ""))
            else:
                if run.get("accuracy") is not None:
                    st.write(f"Accuracy: {float(run['accuracy']):.0%}")
                st.write(f"Errors: {run.get('errors', 0)}")
                st.write(f"Slow rows: {run.get('slow_rows', 0)}")
                if run.get("avg_duration_seconds") is not None:
                    st.write(f"Avg row time: {float(run['avg_duration_seconds']):.1f}s")
                st.write(f"Labels: {run.get('label_counts', {})}")
    if runs and st.button("Clear run history", use_container_width=True):
        clear_run_history()
        st.success("Run history cleared.")


def _model_selector(scope: str) -> str:
    selected = st.selectbox(
        "Model",
        options=_MODEL_OPTIONS,
        index=0,
        help="Choose a model for this run. Default uses MODEL from .env or the project fallback.",
        key=f"{scope}_model_choice",
    )
    if selected == "Default from .env":
        return ""
    if selected == "Custom":
        return st.text_input(
            "Custom model",
            placeholder="provider/model-name, for example ollama/qwen2.5:7b",
            key=f"{scope}_custom_model",
        ).strip()
    return selected


def _resolve_article_input(title: str, article_text: str, article_url: str) -> tuple[str, str]:
    if article_text.strip() or not article_url.strip():
        return title.strip(), article_text.strip()
    with st.spinner("Fetching article text from URL..."):
        fetched = fetch_article_from_url(article_url)
    resolved_title = title.strip() or fetched.title
    st.success(f"Loaded article text from {fetched.url}")
    return resolved_title, fetched.text


def _finalize_batch_output(
    output_row: dict[str, str],
    diagnostic_rows: list[dict[str, str]],
    *,
    batch_run_id: str,
    row_index: int,
    total_rows: int,
    started_at: str,
    start_perf: float,
    phase: str,
    slow_threshold_seconds: float,
    article_text: str,
    error: Exception | None = None,
) -> dict[str, str]:
    finished_at = utc_timestamp()
    duration_seconds = max(time.perf_counter() - start_perf, 0.0)
    is_slow = duration_seconds >= slow_threshold_seconds
    error_details = describe_exception(error)

    output_row.update(
        {
            "batch_run_id": batch_run_id,
            "row_index": str(row_index),
            "duration_seconds": f"{duration_seconds:.2f}",
            "is_slow": str(is_slow),
            "diagnostic_phase": phase,
            "error_type": error_details["error_type"],
        }
    )

    diagnostic_row = {
        "batch_run_id": batch_run_id,
        "row_index": str(row_index),
        "total_rows": str(total_rows),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": f"{duration_seconds:.2f}",
        "analysis_duration_seconds": output_row.get("analysis_duration_seconds", ""),
        "is_slow": str(is_slow),
        "phase": phase,
        "status": output_row.get("status", ""),
        "title": output_row.get("title", ""),
        "url": output_row.get("url", ""),
        "article_chars": str(len(article_text)),
        "expected_label": output_row.get("expected_label", ""),
        "label": output_row.get("label", ""),
        "truth_score": output_row.get("truth_score", ""),
        "confidence": output_row.get("confidence", ""),
        "evidence_mode": output_row.get("evidence_mode", ""),
        "model_name": output_row.get("model_name", ""),
        "analysis_depth": output_row.get("analysis_depth", ""),
        "supporting_source_count": output_row.get("supporting_source_count", ""),
        "opposing_source_count": output_row.get("opposing_source_count", ""),
        "error_type": error_details["error_type"],
        "error": error_details["error"] or output_row.get("error", ""),
        "traceback": error_details["traceback"],
    }
    diagnostic_rows.append(diagnostic_row)
    append_batch_row_log(batch_run_id, diagnostic_row)
    return output_row


def _filter_batch_outputs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []

    st.subheader("Batch Filters")
    labels = _unique_values(rows, "label")
    statuses = _unique_values(rows, "status")
    agreements = _unique_values(rows, "baseline_agreement")

    col1, col2, col3 = st.columns(3)
    selected_statuses = col1.multiselect(
        "Status",
        statuses,
        default=statuses,
    )
    selected_labels = col2.multiselect(
        "Predicted label",
        labels,
        default=labels,
    )
    selected_agreements = col3.multiselect(
        "Baseline agreement",
        agreements,
        default=agreements,
    )

    col4, col5, col6 = st.columns(3)
    min_confidence = col4.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    max_confidence = col5.slider("Maximum confidence", 0.0, 1.0, 1.0, 0.05)
    min_evidence_quality = col6.slider("Minimum evidence quality", 0.0, 1.0, 0.0, 0.05)

    correctness_filter = st.selectbox(
        "Expected-label comparison",
        ["All rows", "Correct only", "Incorrect only", "Has expected label", "Missing expected label"],
    )

    filtered = []
    for row in rows:
        if row.get("status", "") not in selected_statuses:
            continue
        if row.get("label", "") not in selected_labels:
            continue
        if row.get("baseline_agreement", "") not in selected_agreements:
            continue
        confidence = _safe_float(row.get("confidence"))
        if confidence is not None and not (min_confidence <= confidence <= max_confidence):
            continue
        evidence_quality = _safe_float(row.get("evidence_quality_score"))
        if evidence_quality is not None and evidence_quality < min_evidence_quality:
            continue
        if not _passes_correctness_filter(row, correctness_filter):
            continue
        filtered.append(row)
    return filtered


def _unique_values(rows: list[dict[str, str]], key: str) -> list[str]:
    values = sorted({row.get(key, "") for row in rows if row.get(key, "")})
    return values or [""]


def _safe_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _passes_correctness_filter(row: dict[str, str], selected_filter: str) -> bool:
    match = row.get("expected_match") or _expected_match(
        row.get("expected_label", ""),
        row.get("label", ""),
    )
    has_expected = match != "NO EXPECTED LABEL"

    if selected_filter == "All rows":
        return True
    if selected_filter == "Has expected label":
        return has_expected
    if selected_filter == "Missing expected label":
        return not has_expected
    if not has_expected:
        return False
    if selected_filter == "Correct only":
        return match == "CORRECT"
    if selected_filter == "Incorrect only":
        return match == "INCORRECT"
    return True


def _truth_score_caption(truth_score: float) -> str:
    if truth_score >= 0.8:
        return "The judge sees the article as strongly credible."
    if truth_score >= 0.6:
        return "The judge leans toward the article being mostly true."
    if truth_score >= 0.4:
        return "The judge sees a mixed or uncertain article."
    if truth_score >= 0.2:
        return "The judge leans toward the article being misleading or false."
    return "The judge sees the article as strongly unreliable."


def _analysis_timeline_rows(report: object) -> list[dict[str, str]]:
    verdict = report.final_verdict
    quality = report.evidence_quality
    baseline = report.baseline_prediction
    contradiction = report.contradiction_report
    return [
        {
            "step": "1. Claim extraction",
            "agent": "Claim Extractor",
            "status": "Completed",
            "key output": f"{len(report.claims.claims)} claim(s) extracted",
            "detail": "; ".join(report.claims.claims[:3]),
        },
        {
            "step": "2. Supporting case",
            "agent": "Counsel for Credibility",
            "status": "Completed",
            "key output": f"{_verdict_count(report.supporting_case, 'SUPPORTED')} supported claim(s)",
            "detail": report.supporting_case.case_summary,
        },
        {
            "step": "3. Opposing case",
            "agent": "Counsel for Skepticism",
            "status": "Completed",
            "key output": f"{_verdict_count(report.opposing_case, 'CONTRADICTED')} contradicted claim(s)",
            "detail": report.opposing_case.case_summary,
        },
        {
            "step": "4. Bias analysis",
            "agent": "Bias Analyst",
            "status": "Completed",
            "key output": f"{report.bias_report.tone}, bias {report.bias_report.bias_score:.0%}",
            "detail": "; ".join(report.bias_report.flags[:4]),
        },
        {
            "step": "5. Evidence quality",
            "agent": "Evidence Scorer",
            "status": "Completed" if quality else "Skipped",
            "key output": f"{quality.grade}, score {quality.score:.0%}" if quality else "No score",
            "detail": " ".join(quality.notes) if quality else "",
        },
        {
            "step": "6. Classifier baseline",
            "agent": "Local Naive Bayes",
            "status": "Completed" if baseline else "Skipped",
            "key output": (
                f"{baseline.label}, confidence {baseline.confidence:.0%}"
                if baseline
                else "Disabled or unavailable"
            ),
            "detail": ", ".join(baseline.top_indicators[:6]) if baseline else "",
        },
        {
            "step": "7. Contradiction analysis",
            "agent": "Contradiction Detector",
            "status": "Completed" if contradiction else "Skipped",
            "key output": (
                f"{contradiction.level}, score {contradiction.score:.0%}"
                if contradiction
                else "No score"
            ),
            "detail": " ".join(contradiction.notes) if contradiction else "",
        },
        {
            "step": "8. Final verdict",
            "agent": "Chief Fact-Checking Judge",
            "status": "Completed",
            "key output": f"{verdict.label}, truth {verdict.truth_score:.0%}, confidence {verdict.confidence:.0%}",
            "detail": verdict.summary,
        },
    ]


def _evidence_table_rows(legal_case: object) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, item in enumerate(getattr(legal_case, "results", []), start=1):
        rows.append(
            {
                "#": str(index),
                "claim": item.claim,
                "verdict": item.verdict,
                "confidence": f"{item.confidence:.0%}",
                "sources": _format_sources(item.source_urls),
                "evidence": item.evidence,
                "reasoning": item.reasoning,
            }
        )
    return rows


def _format_sources(source_urls: list[str]) -> str:
    if not source_urls:
        return "No source URLs"
    return "\n".join(source_urls)


def _verdict_count(legal_case: object, verdict: str) -> int:
    return sum(1 for item in getattr(legal_case, "results", []) if item.verdict == verdict)


def _comparison_row(
    mode: str,
    report: object,
    *,
    analysis_depth: str,
    model_name: str,
) -> dict[str, str]:
    verdict = report.final_verdict
    baseline = report.baseline_prediction
    quality = report.evidence_quality
    contradiction = report.contradiction_report
    return {
        "mode": mode,
        "analysis_depth": analysis_depth,
        "model_name": model_name or "default",
        "status": "OK",
        "label": verdict.label,
        "truth_score": f"{verdict.truth_score:.4f}",
        "confidence": f"{verdict.confidence:.4f}",
        "bias_score": f"{report.bias_report.bias_score:.4f}",
        "baseline_label": baseline.label if baseline else "",
        "baseline_agreement": (
            _baseline_agreement(verdict.label, baseline.label)
            if baseline
            else ""
        ),
        "evidence_quality_score": f"{quality.score:.4f}" if quality else "",
        "evidence_quality_grade": quality.grade if quality else "",
        "contradiction_score": f"{contradiction.score:.4f}" if contradiction else "",
        "contradiction_level": contradiction.level if contradiction else "",
        "supporting_sources": str(_source_count(report.supporting_case)),
        "opposing_sources": str(_source_count(report.opposing_case)),
        "summary": verdict.summary,
    }


def _source_count(legal_case: object) -> int:
    results = getattr(legal_case, "results", [])
    return sum(len(getattr(item, "source_urls", [])) for item in results)


def _render_batch_metrics(rows: list[dict[str, str]]) -> None:
    completed_rows = [row for row in rows if row.get("status") == "OK"]
    if not rows:
        return

    with st.container(border=True):
        st.subheader("Run Summary")
        st.caption("High-level completion status and average judge scores.")
        successful = len(completed_rows)
        errored = sum(1 for row in rows if row.get("status") == "ERROR")
        skipped = sum(1 for row in rows if row.get("status") == "SKIPPED")
        avg_truth = (
            sum(float(row["truth_score"]) for row in completed_rows) / successful
            if successful
            else 0.0
        )
        avg_confidence = (
            sum(float(row["confidence"]) for row in completed_rows) / successful
            if successful
            else 0.0
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Completed", successful)
        col2.metric("Errored", errored)
        col3.metric("Skipped", skipped)
        col4.metric("Avg truth", f"{avg_truth:.2f}")
        col5.metric("Avg confidence", f"{avg_confidence:.2f}")

    if not completed_rows:
        return

    quality_rows = [
        row for row in completed_rows
        if row.get("evidence_quality_score")
    ]
    baseline_rows = [
        row for row in completed_rows
        if row.get("baseline_agreement") in {"AGREE", "DISAGREE"}
    ]
    if quality_rows or baseline_rows:
        with st.container(border=True):
            st.subheader("Evidence And Baseline Quality")
            st.caption("Quality and agreement signals for completed rows.")
            quality_col, baseline_col = st.columns(2)
            if quality_rows:
                avg_quality = (
                    sum(float(row["evidence_quality_score"]) for row in quality_rows)
                    / len(quality_rows)
                )
                quality_col.metric("Avg evidence quality", f"{avg_quality:.0%}")
            else:
                quality_col.metric("Avg evidence quality", "N/A")
            if baseline_rows:
                agreement_rate = (
                    sum(1 for row in baseline_rows if row["baseline_agreement"] == "AGREE")
                    / len(baseline_rows)
                )
                baseline_col.metric("Agent/baseline agreement", f"{agreement_rate:.0%}")
            else:
                baseline_col.metric("Agent/baseline agreement", "N/A")

    with st.container(border=True):
        st.subheader("Predicted Label Distribution")
        st.caption("Predicted label counts for completed rows.")
        st.bar_chart(_label_distribution_rows(completed_rows), x="label", y="count")

    labeled_rows = [
        row for row in completed_rows
        if row.get("expected_label") in {"REAL", "FAKE", "MIXED", "UNVERIFIABLE", "true", "false"}
    ]
    if not labeled_rows:
        with st.container(border=True):
            st.subheader("Expected Label Evaluation")
            st.info("No expected labels were provided, so accuracy and confusion metrics were skipped.")
        return

    normalized_pairs = [
        (_normalize_expected_label(row["expected_label"]), row["label"])
        for row in labeled_rows
    ]
    with st.container(border=True):
        st.subheader("Expected Label Evaluation")
        st.caption("Accuracy and confusion metrics for rows that include an expected label.")
        exact_matches = sum(1 for expected, predicted in normalized_pairs if expected == predicted)
        accuracy = exact_matches / len(normalized_pairs)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Exact label accuracy", f"{accuracy:.0%}")

        per_class = _per_class_metrics(normalized_pairs)
        metric_cols[1].metric("REAL recall", _format_metric_pct(per_class["REAL"]["recall"]))
        metric_cols[2].metric("FAKE recall", _format_metric_pct(per_class["FAKE"]["recall"]))
        metric_cols[3].metric("Macro recall", _format_metric_pct(_macro_recall(per_class)))

        st.caption("Confusion matrix counts.")
        confusion_rows = _pairs_to_confusion_rows(normalized_pairs)
        st.dataframe(confusion_rows, use_container_width=True)

        with st.expander("Show confusion chart"):
            st.bar_chart(_confusion_chart_rows(normalized_pairs), x="cell", y="count", color="series")

        binary_pairs = _binary_pairs(normalized_pairs)
        if binary_pairs:
            st.divider()
            st.caption("Binary metrics collapse MIXED and UNVERIFIABLE when they map cleanly to REAL or FAKE.")
            binary_metrics = _binary_metrics(binary_pairs)
            binary_cols = st.columns(5)
            binary_cols[0].metric("Binary accuracy", _format_metric_pct(binary_metrics["accuracy"]))
            binary_cols[1].metric("Fake precision", _format_metric_pct(binary_metrics["fake_precision"]))
            binary_cols[2].metric("Fake recall", _format_metric_pct(binary_metrics["fake_recall"]))
            binary_cols[3].metric("Real precision", _format_metric_pct(binary_metrics["real_precision"]))
            binary_cols[4].metric("Real recall", _format_metric_pct(binary_metrics["real_recall"]))


def _render_batch_diagnostics(
    rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    batch_run_id: str,
) -> None:
    if not diagnostic_rows:
        return

    with st.container(border=True):
        st.subheader("Batch Diagnostics")
        _render_batch_diagnostics_content(rows, diagnostic_rows, batch_run_id)


def _render_batch_diagnostics_content(
    rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
    batch_run_id: str,
) -> None:
    durations = [
        float(row["duration_seconds"])
        for row in diagnostic_rows
        if row.get("duration_seconds")
    ]
    slow_rows = [row for row in diagnostic_rows if row.get("is_slow") == "True"]
    failed_rows = [row for row in diagnostic_rows if row.get("status") == "ERROR"]
    slowest_rows = sorted(
        diagnostic_rows,
        key=lambda row: float(row.get("duration_seconds") or 0.0),
        reverse=True,
    )[:5]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows logged", len(rows))
    col2.metric("Slow rows", len(slow_rows))
    col3.metric("Failed rows", len(failed_rows))
    col4.metric("Slowest row", f"{max(durations):.1f}s" if durations else "0.0s")

    diagnostic_preview = [
        {
            "row_index": row.get("row_index", ""),
            "status": row.get("status", ""),
            "duration_seconds": row.get("duration_seconds", ""),
            "analysis_duration_seconds": row.get("analysis_duration_seconds", ""),
            "is_slow": row.get("is_slow", ""),
            "phase": row.get("phase", ""),
            "label": row.get("label", ""),
            "error_type": row.get("error_type", ""),
            "error": row.get("error", ""),
            "title": row.get("title", ""),
        }
        for row in slowest_rows
    ]
    st.caption("Slowest rows and failures are the first place to look when a batch run stalls.")
    st.dataframe(diagnostic_preview, use_container_width=True)

    if failed_rows:
        with st.expander("Failed row details"):
            st.dataframe(
                [
                    {
                        "row_index": row.get("row_index", ""),
                        "phase": row.get("phase", ""),
                        "error_type": row.get("error_type", ""),
                        "error": row.get("error", ""),
                        "title": row.get("title", ""),
                    }
                    for row in failed_rows
                ],
                use_container_width=True,
            )

    st.caption(f"Local JSONL log: {batch_log_path(batch_run_id)}")
    st.download_button(
        "Download diagnostic log CSV",
        data=_rows_to_csv(diagnostic_rows),
        file_name=batch_log_filename(batch_run_id),
        mime="text/csv",
        use_container_width=True,
    )


def _render_batch_results_table(
    outputs: list[dict[str, str]],
    filtered_outputs: list[dict[str, str]],
) -> None:
    with st.container(border=True):
        st.subheader("Results Table And Downloads")
        st.caption(f"Showing {len(filtered_outputs)} of {len(outputs)} row(s).")
        st.dataframe(filtered_outputs, use_container_width=True)
        download_col1, download_col2 = st.columns(2)
        download_col1.download_button(
            "Download all results CSV",
            data=_rows_to_csv(outputs),
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        download_col2.download_button(
            "Download filtered results CSV",
            data=_rows_to_csv(filtered_outputs),
            file_name="filtered_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _pairs_to_confusion_rows(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    labels = ["REAL", "FAKE", "MIXED", "UNVERIFIABLE"]
    rows: list[dict[str, str]] = []
    for expected in labels:
        row: dict[str, str] = {"expected": expected}
        for predicted in labels:
            row[predicted] = str(sum(1 for exp, pred in pairs if exp == expected and pred == predicted))
        rows.append(row)
    return rows


def _label_distribution_rows(rows: list[dict[str, str]]) -> list[dict[str, int | str]]:
    labels = ["REAL", "FAKE", "MIXED", "UNVERIFIABLE", "ERROR", "SKIPPED"]
    return [
        {
            "label": label,
            "count": sum(1 for row in rows if row.get("label") == label),
        }
        for label in labels
    ]


def _confusion_chart_rows(pairs: list[tuple[str, str]]) -> list[dict[str, int | str]]:
    labels = ["REAL", "FAKE", "MIXED", "UNVERIFIABLE"]
    chart_rows: list[dict[str, int | str]] = []
    for expected in labels:
        for predicted in labels:
            chart_rows.append(
                {
                    "cell": f"{expected}->{predicted}",
                    "series": expected,
                    "count": sum(
                        1 for exp, pred in pairs if exp == expected and pred == predicted
                    ),
                }
            )
    return chart_rows


def _per_class_metrics(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float | None]]:
    labels = ["REAL", "FAKE", "MIXED", "UNVERIFIABLE"]
    metrics: dict[str, dict[str, float | None]] = {}
    for label in labels:
        true_positive = sum(1 for exp, pred in pairs if exp == label and pred == label)
        predicted_positive = sum(1 for _, pred in pairs if pred == label)
        actual_positive = sum(1 for exp, _ in pairs if exp == label)
        precision = true_positive / predicted_positive if predicted_positive else None
        recall = true_positive / actual_positive if actual_positive else None
        metrics[label] = {"precision": precision, "recall": recall}
    return metrics


def _macro_recall(per_class: dict[str, dict[str, float | None]]) -> float | None:
    recalls = [
        values["recall"]
        for values in per_class.values()
        if values["recall"] is not None
    ]
    if not recalls:
        return None
    return sum(recalls) / len(recalls)


def _binary_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    mapped_pairs: list[tuple[str, str]] = []
    for expected, predicted in pairs:
        mapped_expected = _binary_label(expected)
        mapped_predicted = _binary_label(predicted)
        if mapped_expected is None or mapped_predicted is None:
            continue
        mapped_pairs.append((mapped_expected, mapped_predicted))
    return mapped_pairs


def _binary_label(label: str) -> str | None:
    if label == "REAL":
        return "REAL"
    if label == "FAKE":
        return "FAKE"
    return None


def _binary_metrics(pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    total = len(pairs)
    correct = sum(1 for expected, predicted in pairs if expected == predicted)

    fake_tp = sum(1 for expected, predicted in pairs if expected == "FAKE" and predicted == "FAKE")
    fake_predicted = sum(1 for _, predicted in pairs if predicted == "FAKE")
    fake_actual = sum(1 for expected, _ in pairs if expected == "FAKE")

    real_tp = sum(1 for expected, predicted in pairs if expected == "REAL" and predicted == "REAL")
    real_predicted = sum(1 for _, predicted in pairs if predicted == "REAL")
    real_actual = sum(1 for expected, _ in pairs if expected == "REAL")

    return {
        "accuracy": correct / total if total else None,
        "fake_precision": fake_tp / fake_predicted if fake_predicted else None,
        "fake_recall": fake_tp / fake_actual if fake_actual else None,
        "real_precision": real_tp / real_predicted if real_predicted else None,
        "real_recall": real_tp / real_actual if real_actual else None,
    }


def _format_metric_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0%}"


if __name__ == "__main__":
    main()
