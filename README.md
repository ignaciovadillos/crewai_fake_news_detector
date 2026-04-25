# Detector Fake News

This project is a simple, teachable fake-news detector built around a CrewAI
crew and a small Streamlit UI.

## What this MVP does

It analyzes one article through a sequential, courtroom-style workflow:

1. `Claim Extractor` pulls out 3-5 factual claims.
2. `Counsel for Credibility` looks for evidence that supports the article.
3. `Counsel for Skepticism` looks for evidence that challenges the article.
4. `Bias Analyst` scores the tone and framing.
5. `Judge` combines all prior outputs into a final REAL or FAKE verdict.


## Why this architecture

- It matches the assignment's recommendation for a sequential pipeline.
- It is easy to inspect because each stage produces a structured output.
- It leaves a clean path for later upgrades:
  - parallel support vs. challenge research
  - dataset evaluation
  - Hugging Face classifier as an extra signal
  - real Tavily/Serper search

## Project layout

```text
src/detector_fake_news/
├── config/
│   ├── agents.yaml
│   └── tasks.yaml
├── tools/
│   └── research.py
├── crew.py
├── llm.py
├── main.py
├── models.py
├── service.py
└── ui.py
```

## Setup

Install dependencies:

```bash
uv sync
```

Optional environment variables:

```bash
# OpenAI option
OPENAI_API_KEY=...
MODEL=openai/gpt-4o-mini

# Or local Ollama option
MODEL=ollama/qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

If `MODEL` is not set, the app defaults to:

- `openai/gpt-4o-mini` when `OPENAI_API_KEY` exists
- otherwise `ollama/qwen2.5:7b`

## Run

CLI example:

```bash
uv run detector-fake-news
```

Streamlit UI:

```bash
uv run streamlit run src/detector_fake_news/ui.py
```

## How the Crew works

### Agents

Agents are the personalities and responsibilities:

- `claim_extractor`: turns a long article into a shortlist of checkable claims
- `supporting_counsel`: builds the strongest honest case for credibility
- `opposing_counsel`: builds the strongest honest case against credibility
- `bias_analyst`: ignores truth value and focuses on rhetoric and framing
- `judge`: weighs the evidence and issues the final decision

### Tasks

Tasks are the actual work items assigned to those agents.

- Extract claims from the article
- Build the supporting case
- Build the skeptical case
- Analyze tone and bias
- Issue the verdict

### Workflow

The workflow is the order of task execution inside the crew:

```text
Article -> Claims -> Support case -> Challenge case -> Bias report -> Judge
```

The crew is sequential on purpose for the MVP. It is simpler to understand than
parallel execution, and it matches the assignment's beginner-friendly option.

## Search tool behavior

The app ships with a custom research tool so you are not blocked by missing
`crewai_tools`.

- If `TAVILY_API_KEY` is set, it uses Tavily search.
- Otherwise it falls back to Wikipedia search and summaries.

That means the project works as a base without extra package installation, while
still giving you a clean upgrade path for better evidence gathering later.
