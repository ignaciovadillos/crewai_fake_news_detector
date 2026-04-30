"""Fetch and extract readable article text from URLs."""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse


@dataclass(frozen=True)
class FetchedArticle:
    title: str
    text: str
    url: str


class _ArticleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self._tag_stack: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg", "nav", "footer"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg", "nav", "footer"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str):
        cleaned = _normalize_whitespace(data)
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
            return
        if self._skip_depth:
            return
        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag in {"p", "h1", "h2", "h3", "li", "blockquote", "article", "main"}:
            self.body_parts.append(cleaned)


def fetch_article_from_url(url: str, *, timeout: int = 20) -> FetchedArticle:
    """Fetch a URL and extract a simple readable title/text payload."""
    normalized_url = _normalize_url(url)
    request = urllib.request.Request(
        normalized_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; "
                "+https://localhost)"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        raw = response.read(2_000_000)

    if "html" not in content_type.lower() and not normalized_url.lower().endswith((".html", ".htm")):
        text = raw.decode(_encoding_from_content_type(content_type), errors="replace")
        return FetchedArticle(title=_fallback_title(normalized_url), text=_normalize_whitespace(text), url=normalized_url)

    html = raw.decode(_encoding_from_content_type(content_type), errors="replace")
    parser = _ArticleHTMLParser()
    parser.feed(html)

    title = _clean_title(" ".join(parser.title_parts)) or _fallback_title(normalized_url)
    text = _normalize_article_text(parser.body_parts)
    if not text:
        text = _fallback_text_from_html(html)
    if not text:
        raise ValueError("Could not extract readable article text from the URL.")

    return FetchedArticle(title=title, text=text, url=normalized_url)


def _normalize_url(url: str) -> str:
    stripped = url.strip()
    if not stripped:
        raise ValueError("URL is empty.")
    parsed = urlparse(stripped)
    if not parsed.scheme:
        stripped = f"https://{stripped}"
        parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be an http or https address.")
    return stripped


def _encoding_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([^;]+)", content_type, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "utf-8"


def _normalize_article_text(parts: list[str]) -> str:
    paragraphs = []
    seen: set[str] = set()
    for part in parts:
        cleaned = _normalize_whitespace(part)
        if len(cleaned) < 40:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        paragraphs.append(cleaned)
    return "\n\n".join(paragraphs)


def _fallback_text_from_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _normalize_whitespace(text)


def _clean_title(title: str) -> str:
    cleaned = _normalize_whitespace(title)
    if " | " in cleaned:
        cleaned = cleaned.split(" | ")[0]
    if " - " in cleaned and len(cleaned) > 90:
        cleaned = cleaned.split(" - ")[0]
    return cleaned


def _fallback_title(url: str) -> str:
    parsed = urlparse(url)
    path_part = parsed.path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    return path_part.title() if path_part else parsed.netloc


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
