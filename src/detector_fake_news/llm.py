"""LLM configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """Return the project LLM based on environment variables."""
    model = os.getenv("MODEL")
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
