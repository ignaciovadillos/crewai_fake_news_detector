"""Research tool with Tavily and Wikipedia fallback."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ArticleResearchInput(BaseModel):
    query: str = Field(..., description="Search query for evidence gathering.")
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of results to return.",
    )


class ArticleResearchTool(BaseTool):
    name: str = "article_research"
    description: str = (
        "Searches for evidence about a claim. Uses Tavily when TAVILY_API_KEY is "
        "available, otherwise falls back to Wikipedia search and summaries."
    )
    args_schema: Type[BaseModel] = ArticleResearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            return self._run_tavily(query=query, max_results=max_results, api_key=tavily_api_key)
        return self._run_wikipedia(query=query, max_results=max_results)

    def _run_tavily(self, query: str, max_results: int, api_key: str) -> str:
        payload = json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            return json.dumps(
                {
                    "query": query,
                    "error": f"Tavily search failed: {exc}",
                    "results": [],
                },
                indent=2,
            )

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in data.get("results", [])[:max_results]
        ]
        return json.dumps(
            {
                "query": query,
                "answer": data.get("answer", ""),
                "results": results,
            },
            indent=2,
        )

    def _run_wikipedia(self, query: str, max_results: int) -> str:
        search_params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
            }
        )
        search_url = f"https://en.wikipedia.org/w/api.php?{search_params}"
        try:
            with urllib.request.urlopen(search_url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            return json.dumps(
                {
                    "query": query,
                    "error": f"Wikipedia search failed: {exc}",
                    "results": [],
                },
                indent=2,
            )

        pages = data.get("query", {}).get("search", [])[:max_results]
        results: list[dict[str, Any]] = []
        for page in pages:
            title = page.get("title", "")
            summary_url = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{urllib.parse.quote(title)}"
            )
            try:
                with urllib.request.urlopen(summary_url, timeout=20) as response:
                    summary = json.loads(response.read().decode("utf-8"))
            except Exception:
                summary = {}

            results.append(
                {
                    "title": title,
                    "url": summary.get(
                        "content_urls",
                        {},
                    ).get("desktop", {}).get("page", ""),
                    "content": summary.get("extract", ""),
                }
            )

        return json.dumps(
            {
                "query": query,
                "source": "wikipedia_fallback",
                "results": results,
            },
            indent=2,
        )
