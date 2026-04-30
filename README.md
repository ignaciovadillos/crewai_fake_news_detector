# Multi-Agent Fake News Detector

This project is a CrewAI and Streamlit fake-news analysis app. It analyzes news
articles with a courtroom-style multi-agent workflow, supports online and
offline evidence modes, and includes batch evaluation tools for testing against
CSV datasets.

The current version is designed to run locally with Ollama by default, while
still supporting OpenAI-compatible model strings when API keys are available.

## Main Features

- Single article analysis with claim extraction, supporting evidence, opposing evidence, bias analysis, and a final judge.
- Batch CSV analysis for datasets with `article_text` or `text`, optional `url` or `article_url`, and optional expected labels.
- Evidence modes:
  - `online`: searches public sources through Google Fact Check, Tavily trusted domains, GDELT, and Wikipedia fallback.
  - `offline`: searches the local Kaggle-style `archive/fake.csv` and `archive/true.csv` dataset for similarity and style evidence.
  - `hybrid`: combines offline dataset evidence with online evidence.
- Local memory stored in `.runtime/analysis_memory.jsonl` for consistency across repeated analyses.
- Local Naive Bayes baseline classifier trained from the `archive` dataset.
- Evidence quality scoring, contradiction detection, citation enforcement, and automatic final-verdict calibration.
- URL ingestion for single article and batch workflows.
- Analysis depth selector: `Quick`, `Standard`, and `Deep`.
- Model selector for default, Ollama, OpenAI, or custom `provider/model` values.
- Batch filters, expected-label comparison, metrics, confusion tables, and downloadable result CSVs.
- Batch diagnostics for slow or failed rows, including row timings, failure phase, error type, and downloadable diagnostic CSV.
- Markdown report download for single-article runs.
- Persistent run history stored under `.runtime/run_history.jsonl`.
- Unit tests for helper logic, diagnostics, evidence quality, contradiction handling, and async CrewAI output handling.

## Project Layout

```text
src/detector_fake_news/
|-- config/
|   |-- agents.yaml
|   `-- tasks.yaml
|-- tools/
|   `-- research.py
|-- article_fetcher.py
|-- classifier.py
|-- contradictions.py
|-- crew.py
|-- diagnostics.py
|-- evidence_quality.py
|-- history.py
|-- llm.py
|-- main.py
|-- memory.py
|-- models.py
|-- reporting.py
|-- runtime.py
|-- service.py
|-- ui.py
`-- ui_helpers.py

tests/
|-- test_diagnostics.py
|-- test_helpers.py
|-- test_service_outputs.py
`-- test_ui_helpers.py
```

## Setup

Python `>=3.10,<3.14` is required.

Recommended install from the project root:

```powershell
python -m pip install -e .
```

Dependency-only install:

```powershell
python -m pip install -r requirements.txt
```

If using the dependency-only install, run commands from the project root and set
`PYTHONPATH=src` first:

```powershell
$env:PYTHONPATH = "src"
```

## Environment

Copy `.env.example` to `.env` and adjust as needed.

Default local Ollama setup:

```env
MODEL=ollama/qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

Optional OpenAI setup:

```env
OPENAI_API_KEY=sk-...
MODEL=openai/gpt-4o-mini
```

Optional search provider keys:

```env
TAVILY_API_KEY=tvly-...
GOOGLE_FACTCHECK_API_KEY=...
```

If `MODEL` is not set, the app uses `openai/gpt-4o-mini` when
`OPENAI_API_KEY` exists; otherwise it falls back to `ollama/qwen2.5:7b`.

CrewAI and app state are redirected into `.runtime/` by `runtime.py`, including
CrewAI storage, local memory, run history, and batch diagnostics. This avoids
Windows `AppData` permission issues in restricted environments.

## Run The App

Streamlit UI:

```powershell
python -m streamlit run src/detector_fake_news/ui.py
```

CLI smoke analysis:

```powershell
python -m detector_fake_news.main
```

If the CLI import cannot find the package after a dependency-only install:

```powershell
$env:PYTHONPATH = "src"
python -m detector_fake_news.main
```

## CSV Inputs

Batch mode accepts CSV files with:

- `article_text` or `text` for article body text.
- `url` or `article_url` when the app should fetch article text.
- `title` for the headline.
- `expected_label`, `ground_truth`, or `label` for optional evaluation.

Sample files are included:

- `sample_batch_2_rows.csv`
- `sample_batch_archive.csv`

For offline mode, place the Kaggle-style dataset in:

```text
archive/fake.csv
archive/true.csv
```

Offline mode is a similarity and style signal only. It is not live factual
verification.

## Workflow

The CrewAI workflow has five agents:

1. `Investigative Claim Extractor` extracts concise factual claims.
2. `Counsel for Credibility` builds the supporting case.
3. `Counsel for Skepticism` builds the opposing case.
4. `Media Bias Analyst` scores tone and framing.
5. `Chief Fact-Checking Judge` issues the final verdict.

The claim extraction runs first. Supporting and opposing evidence tasks then run
asynchronously for speed. Bias analysis and the judge run after the evidence
stage.

Because asynchronous CrewAI task outputs can arrive out of order, the service
maps structured outputs by schema rather than by list position. It also recovers
claim lists from evidence items if CrewAI omits the claim-extraction structured
output in `tasks_output`.

## Scoring Notes

- `label`: final categorical verdict, one of `REAL`, `FAKE`, `MIXED`, or `UNVERIFIABLE`.
- `truth_score`: estimated truthfulness from `0.0` to `1.0`.
- `confidence`: certainty in the assessment, separate from truthfulness.
- `bias_score`: rhetorical or framing bias from `0.0` to `1.0`.
- Evidence quality considers citations, source counts, trusted domains, and unresolved claims.
- Supporting claims marked `SUPPORTED` without source URLs are downgraded and penalized in the final verdict.
- Contradiction detection reduces the verdict score when supporting and opposing cases contain strong conflicts.

## Batch Diagnostics

Batch runs include diagnostic columns and a separate diagnostic CSV:

- `batch_run_id`
- `row_index`
- `duration_seconds`
- `analysis_duration_seconds`
- `is_slow`
- `diagnostic_phase`
- `error_type`
- `error`

The Streamlit UI also stores JSONL diagnostics under:

```text
.runtime/batch_diagnostics/
```

Use the slow-row threshold in the batch UI to flag rows that take longer than
expected.

## Tests

Run the test suite:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Compile check:

```powershell
python -m compileall src tests
```

The latest verified suite contains `17` passing tests.

## Last Verified Runtime

The project was last smoke-tested with:

- `crewai 1.14.3`
- `streamlit 1.57.0`
- `pydantic 2.11.10`
- Local Ollama model path: `ollama/qwen2.5:7b`

Verified checks included unit tests, compile checks, app imports, Streamlit
headless startup, crew assembly, baseline classifier smoke test, and a full live
Ollama CLI analysis.
