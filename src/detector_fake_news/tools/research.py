"""Research tool with online and local dataset-backed evidence modes."""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

EvidenceMode = Literal["online", "offline", "hybrid"]

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because",
    "been", "before", "being", "between", "both", "but", "can", "could",
    "did", "does", "for", "from", "had", "has", "have", "into", "its",
    "more", "not", "now", "off", "one", "only", "over", "said", "say",
    "says", "she", "should", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "too",
    "under", "was", "were", "what", "when", "where", "which", "while",
    "who", "will", "with", "would", "you", "your",
}
_SENSATIONAL_TERMS = {
    "bombshell", "breaking", "corrupt", "destroy", "disaster", "exposed",
    "fake", "furious", "horrifying", "humiliated", "insane", "lies",
    "massive", "outrage", "scandal", "secret", "shocking", "slam",
    "stunning", "terrifying", "unbelievable",
}
_VAGUE_ATTRIBUTION_TERMS = {
    "allegedly", "anonymous", "insiders", "many people", "reportedly",
    "rumor", "sources say", "some say", "unnamed",
}
_TRUSTED_SEARCH_DOMAINS = [
    "apnews.com",
    "bbc.com",
    "factcheck.afp.com",
    "factcheck.org",
    "fullfact.org",
    "leadstories.com",
    "politifact.com",
    "reuters.com",
    "snopes.com",
]


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
        "Searches for evidence about a claim. In online mode it uses Tavily or "
        "Wikipedia. In offline mode it searches the local fake/real news dataset. "
        "Hybrid mode returns both local and online evidence."
    )
    args_schema: Type[BaseModel] = ArticleResearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        rewritten_queries = _rewrite_query(query)
        primary_query = rewritten_queries[0]
        mode = _evidence_mode()
        if mode == "offline":
            return self._run_offline(
                query=primary_query,
                max_results=max_results,
                original_query=query,
                rewritten_queries=rewritten_queries,
            )
        if mode == "hybrid":
            return self._run_hybrid(
                query=primary_query,
                max_results=max_results,
                original_query=query,
                rewritten_queries=rewritten_queries,
            )
        return self._run_online(
            query=primary_query,
            max_results=max_results,
            original_query=query,
            rewritten_queries=rewritten_queries,
        )

    def _run_online(
        self,
        query: str,
        max_results: int,
        *,
        original_query: str | None = None,
        rewritten_queries: list[str] | None = None,
    ) -> str:
        rewritten_queries = rewritten_queries or [query]
        results: list[dict[str, Any]] = []
        provider_errors: list[str] = []

        fact_check_results, fact_check_error = self._run_google_fact_check(
            query=rewritten_queries[0],
            max_results=max_results,
        )
        results.extend(fact_check_results)
        if fact_check_error:
            provider_errors.append(fact_check_error)

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_results, tavily_error = self._run_tavily(
                query=rewritten_queries[0],
                max_results=max_results,
                api_key=tavily_api_key,
                include_domains=_TRUSTED_SEARCH_DOMAINS,
            )
            results.extend(tavily_results)
            if tavily_error:
                provider_errors.append(tavily_error)

        for gdelt_query in rewritten_queries[:2]:
            gdelt_results, gdelt_error = self._run_gdelt(query=gdelt_query, max_results=max_results)
            results.extend(gdelt_results)
            if gdelt_error:
                provider_errors.append(gdelt_error)

        results = _dedupe_results(results)[:max_results]
        if results:
            return json.dumps(
                {
                    "query": original_query or query,
                    "rewritten_queries": rewritten_queries,
                    "mode": "online",
                    "source": "fact_check_and_trusted_news_search",
                    "providers": ["google_fact_check", "tavily_trusted_domains", "gdelt"],
                    "provider_errors": provider_errors,
                    "results": results,
                },
                indent=2,
            )

        wikipedia_payload = json.loads(self._run_wikipedia(query=rewritten_queries[0], max_results=max_results))
        wikipedia_payload["query"] = original_query or query
        wikipedia_payload["rewritten_queries"] = rewritten_queries
        if provider_errors:
            wikipedia_payload["provider_errors"] = provider_errors
        return json.dumps(wikipedia_payload, indent=2)

    def _run_hybrid(
        self,
        query: str,
        max_results: int,
        *,
        original_query: str | None = None,
        rewritten_queries: list[str] | None = None,
    ) -> str:
        rewritten_queries = rewritten_queries or [query]
        offline_payload = json.loads(
            self._run_offline(
                query=rewritten_queries[0],
                max_results=max_results,
                original_query=original_query or query,
                rewritten_queries=rewritten_queries,
            )
        )
        online_payload = json.loads(
            self._run_online(
                query=rewritten_queries[0],
                max_results=max_results,
                original_query=original_query or query,
                rewritten_queries=rewritten_queries,
            )
        )
        return json.dumps(
            {
                "query": original_query or query,
                "rewritten_queries": rewritten_queries,
                "mode": "hybrid",
                "offline": offline_payload,
                "online": online_payload,
            },
            indent=2,
        )

    def _run_offline(
        self,
        query: str,
        max_results: int,
        *,
        original_query: str | None = None,
        rewritten_queries: list[str] | None = None,
    ) -> str:
        rewritten_queries = rewritten_queries or [query]
        corpus = _load_offline_corpus()
        query_tokens = _tokenize(query)
        if not corpus:
            return json.dumps(
                {
                    "query": original_query or query,
                    "rewritten_queries": rewritten_queries,
                    "mode": "offline",
                    "error": "No local dataset found. Expected archive/fake.csv and archive/true.csv.",
                    "results": [],
                    "features": _text_features(query),
                },
                indent=2,
            )

        scored: list[tuple[float, dict[str, Any]]] = []
        for item in corpus:
            overlap = query_tokens.intersection(item["tokens"])
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append((score, item | {"matched_terms": sorted(overlap)[:12]}))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [
            {
                "title": item["title"],
                "label": item["label"],
                "subject": item["subject"],
                "date": item["date"],
                "url": item["url"],
                "similarity": round(score, 4),
                "matched_terms": item["matched_terms"],
                "content": item["excerpt"],
            }
            for score, item in scored[:max_results]
        ]

        label_counts = {
            "fake": sum(1 for result in results if result["label"] == "fake"),
            "true": sum(1 for result in results if result["label"] == "true"),
        }
        return json.dumps(
            {
                "query": original_query or query,
                "rewritten_queries": rewritten_queries,
                "mode": "offline",
                "source": "local_kaggle_archive",
                "dataset_size": len(corpus),
                "result_label_counts": label_counts,
                "interpretation_note": (
                    "Offline results show similarity to labeled dataset examples. "
                    "They are style and precedent signals, not live factual verification."
                ),
                "features": _text_features(query),
                "results": results,
            },
            indent=2,
        )

    def _run_google_fact_check(
        self,
        query: str,
        max_results: int,
    ) -> tuple[list[dict[str, Any]], str]:
        api_key = os.getenv("GOOGLE_FACTCHECK_API_KEY") or os.getenv("FACTCHECK_API_KEY")
        if not api_key:
            return [], "Google Fact Check skipped: GOOGLE_FACTCHECK_API_KEY is not set."

        params = urllib.parse.urlencode(
            {
                "query": query,
                "languageCode": "en",
                "pageSize": max_results,
                "key": api_key,
            }
        )
        request_url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?{params}"
        try:
            with urllib.request.urlopen(request_url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network/API-key dependent
            return [], f"Google Fact Check failed: {exc}"

        results: list[dict[str, Any]] = []
        for claim in data.get("claims", [])[:max_results]:
            reviews = claim.get("claimReview", []) or []
            review = reviews[0] if reviews else {}
            publisher = review.get("publisher", {}) or {}
            title = review.get("title") or claim.get("text", "")
            rating = review.get("textualRating", "")
            results.append(
                {
                    "title": title,
                    "url": review.get("url", ""),
                    "content": (
                        f"Claim: {claim.get('text', '')}\n"
                        f"Rating: {rating}\n"
                        f"Publisher: {publisher.get('name', '')}"
                    ).strip(),
                    "provider": "google_fact_check",
                    "source_type": "fact_check",
                    "rating": rating,
                    "publisher": publisher.get("name", ""),
                }
            )
        return results, ""

    def _run_gdelt(
        self,
        query: str,
        max_results: int,
    ) -> tuple[list[dict[str, Any]], str]:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": max_results,
                "sort": "HybridRel",
            }
        )
        request_url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
        try:
            with urllib.request.urlopen(request_url, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            return [], f"GDELT search failed: {exc}"

        results = [
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "content": (
                    f"Domain: {article.get('domain', '')}. "
                    f"Seen date: {article.get('seendate', '')}. "
                    f"Language: {article.get('language', '')}."
                ),
                "provider": "gdelt",
                "source_type": "news_index",
                "domain": article.get("domain", ""),
                "seen_date": article.get("seendate", ""),
            }
            for article in data.get("articles", [])[:max_results]
        ]
        return results, ""

    def _run_tavily(
        self,
        query: str,
        max_results: int,
        api_key: str,
        include_domains: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        payload = json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "topic": "news",
                "max_results": max_results,
                "include_answer": True,
                "include_domains": include_domains or [],
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
            return [], f"Tavily search failed: {exc}"

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "provider": "tavily_trusted_domains",
                "source_type": "trusted_news_or_fact_check",
            }
            for item in data.get("results", [])[:max_results]
        ]
        if data.get("answer"):
            results.insert(
                0,
                {
                    "title": "Tavily answer summary",
                    "url": "",
                    "content": data.get("answer", ""),
                    "provider": "tavily_trusted_domains",
                    "source_type": "answer_summary",
                },
            )
        return results[:max_results], ""

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


def _evidence_mode() -> EvidenceMode:
    mode = os.getenv("DETECTOR_EVIDENCE_MODE", "online").strip().lower()
    if mode in {"offline", "hybrid"}:
        return mode  # type: ignore[return-value]
    return "online"


def _dedupe_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for result in results:
        key = result.get("url") or result.get("title") or json.dumps(result, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _rewrite_query(query: str) -> list[str]:
    normalized = _normalize_query_text(query)
    tokens = _tokenize(normalized)
    quoted_entities = _quoted_entities(query)
    years = _YEAR_PATTERN.findall(query)
    numbers = _NUMBER_PATTERN.findall(query)

    primary_terms = _ordered_unique(
        quoted_entities
        + years
        + numbers
        + [
            token
            for token in tokens
            if len(token) > 3
        ][:10]
    )
    primary = " ".join(primary_terms).strip() or normalized

    variants = [primary]
    if quoted_entities:
        variants.append(" ".join(_ordered_unique(quoted_entities + years + tokens[:6])))
    if years or numbers:
        variants.append(" ".join(_ordered_unique(tokens[:8] + years + numbers)))
    variants.append(normalized)

    return [
        variant
        for variant in _ordered_unique(variants)
        if variant
    ][:3]


def _normalize_query_text(query: str) -> str:
    cleaned = query.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(
        r"^(claim|article|headline|the article says|the article claims)\s*[:\-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned[:240]


def _quoted_entities(text: str) -> list[str]:
    capitalized_phrases = re.findall(
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b",
        text,
    )
    return [
        phrase
        for phrase in capitalized_phrases
        if phrase.lower() not in _STOPWORDS and len(phrase) > 2
    ][:6]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique_values.append(normalized)
    return unique_values


@lru_cache(maxsize=1)
def _load_offline_corpus() -> tuple[dict[str, Any], ...]:
    project_root = Path(__file__).resolve().parents[3]
    archive_dir = project_root / "archive"
    max_rows_per_file = int(os.getenv("DETECTOR_OFFLINE_MAX_ROWS_PER_FILE", "2500"))
    rows: list[dict[str, Any]] = []

    for filename, label in [("fake.csv", "fake"), ("true.csv", "true")]:
        path = archive_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader, start=1):
                if index > max_rows_per_file:
                    break
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                combined_text = f"{title} {text}".strip()
                rows.append(
                    {
                        "title": title,
                        "label": label,
                        "subject": (row.get("subject") or "").strip(),
                        "date": (row.get("date") or "").strip(),
                        "url": f"offline://archive/{filename}#row={index}",
                        "excerpt": text[:700],
                        "tokens": _tokenize(combined_text),
                    }
                )
    return tuple(rows)


def _tokenize(text: str) -> set[str]:
    return {
        word.lower().strip("'")
        for word in _WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }


def _text_features(text: str) -> dict[str, Any]:
    words = _WORD_PATTERN.findall(text)
    lowered = text.lower()
    sensational_hits = sorted(term for term in _SENSATIONAL_TERMS if term in lowered)
    vague_hits = sorted(term for term in _VAGUE_ATTRIBUTION_TERMS if term in lowered)
    all_caps_words = [word for word in text.split() if len(word) > 2 and word.isupper()]
    return {
        "word_count": len(words),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "all_caps_word_count": len(all_caps_words),
        "sensational_terms": sensational_hits,
        "vague_attribution_terms": vague_hits,
    }
