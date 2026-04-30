"""LLM configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from detector_fake_news.runtime import configure_runtime_environment

configure_runtime_environment()

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=16)
def get_llm(model_override: str | None = None) -> LLM:
    """Return the project LLM based on environment variables."""
    model = model_override or os.getenv("MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    if not model:
        if os.getenv("OPENAI_API_KEY"):
            model = "openai/gpt-4o-mini"
        else:
            model = "ollama/qwen2.5:7b"

    if model.startswith("ollama/"):
        return LLM(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )

    return LLM(
        model=model,
        temperature=temperature,
    )


def selected_model_name() -> str:
    """Return the model name that would be used without a runtime override."""
    model = os.getenv("MODEL")
    if model:
        return model
    if os.getenv("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    return "ollama/qwen2.5:7b"
