"""Streamlit UI for the fake-news detector."""

from __future__ import annotations

import csv
import io
import json

import streamlit as st

from detector_fake_news.service import analyze_article


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
            "This is the sequential MVP. Later, the supporting and opposing case "
            "tasks can be run in parallel."
        )

    single_tab, batch_tab = st.tabs(["Single Article", "Batch CSV"])

    with single_tab:
        render_single_article()

    with batch_tab:
        render_batch_mode()


def render_single_article() -> None:
    title = st.text_input("Article title", placeholder="Enter the news headline")
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
        if not article_text.strip():
            st.error("Please provide article text before running the analysis.")
            return

        with st.spinner("Running the agentic workflow..."):
            report = analyze_article(title=title, article_text=article_text)

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

        judge_tab, claims_tab, support_tab, oppose_tab, bias_tab = st.tabs(
            ["Judge", "Claims", "Supporting Case", "Opposing Case", "Bias"]
        )

        with judge_tab:
            st.json(verdict.model_dump())

        with claims_tab:
            st.json(report.claims.model_dump())

        with support_tab:
            st.json(report.supporting_case.model_dump())

        with oppose_tab:
            st.json(report.opposing_case.model_dump())

        with bias_tab:
            st.json(report.bias_report.model_dump())


def render_batch_mode() -> None:
    st.write(
        "Upload a CSV with either `article_text` or `text`, and optionally `title`."
    )
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
        progress = st.progress(0.0)

        for index, row in enumerate(rows, start=1):
            title = (row.get("title") or "").strip()
            article_text = (row.get("article_text") or row.get("text") or "").strip()
            if not article_text:
                outputs.append(
                    {
                        "title": title,
                        "label": "SKIPPED",
                        "truth_score": "0.0",
                        "confidence": "0.0",
                        "summary": "Missing article_text/text column value.",
                    }
                )
                progress.progress(index / max(len(rows), 1))
                continue

            report = analyze_article(title=title, article_text=article_text)
            outputs.append(
                {
                    "title": title,
                    "label": report.final_verdict.label,
                    "truth_score": f"{report.final_verdict.truth_score:.4f}",
                    "confidence": f"{report.final_verdict.confidence:.4f}",
                    "summary": report.final_verdict.summary,
                    "claims": json.dumps(report.claims.model_dump(), ensure_ascii=True),
                }
            )
            progress.progress(index / max(len(rows), 1))

        st.success("Batch run finished.")
        st.dataframe(outputs, use_container_width=True)
        st.download_button(
            "Download results CSV",
            data=_rows_to_csv(outputs),
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def _rows_to_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


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


if __name__ == "__main__":
    main()
