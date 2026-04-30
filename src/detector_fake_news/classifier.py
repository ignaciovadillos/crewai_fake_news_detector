"""Local fake/real baseline classifier trained from the Kaggle archive."""

from __future__ import annotations

import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from detector_fake_news.models import BaselinePrediction

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z']+")
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


@dataclass(frozen=True)
class NaiveBayesModel:
    class_counts: dict[str, int]
    token_counts: dict[str, Counter[str]]
    total_tokens: dict[str, int]
    vocabulary: set[str]


def predict_baseline(title: str, article_text: str) -> BaselinePrediction | None:
    """Predict REAL/FAKE with a local bag-of-words Naive Bayes baseline."""
    model = _train_model()
    if not model or not model.vocabulary:
        return None

    tokens = _tokenize(f"{title} {article_text}")
    if not tokens:
        return None

    token_counts = Counter(tokens)
    log_scores = {
        "fake": _class_log_score("fake", token_counts, model),
        "true": _class_log_score("true", token_counts, model),
    }
    probabilities = _softmax(log_scores)
    fake_probability = probabilities["fake"]
    real_probability = probabilities["true"]
    label = "FAKE" if fake_probability >= real_probability else "REAL"
    confidence = max(fake_probability, real_probability)
    indicators = _top_indicators(token_counts, model, predicted_label=label)

    return BaselinePrediction(
        label=label,
        real_probability=real_probability,
        fake_probability=fake_probability,
        confidence=confidence,
        top_indicators=indicators,
        training_examples=sum(model.class_counts.values()),
    )


@lru_cache(maxsize=1)
def _train_model() -> NaiveBayesModel | None:
    archive_dir = _project_root() / "archive"
    max_rows_per_file = int(os.getenv("DETECTOR_BASELINE_MAX_ROWS_PER_FILE", "4000"))
    class_counts = {"fake": 0, "true": 0}
    token_counts: dict[str, Counter[str]] = {"fake": Counter(), "true": Counter()}
    total_tokens = {"fake": 0, "true": 0}
    vocabulary: set[str] = set()

    for filename, label in [("fake.csv", "fake"), ("true.csv", "true")]:
        path = archive_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader, start=1):
                if index > max_rows_per_file:
                    break
                text = f"{row.get('title') or ''} {row.get('text') or ''}"
                tokens = _tokenize(text)
                if not tokens:
                    continue
                class_counts[label] += 1
                token_counts[label].update(tokens)
                total_tokens[label] += len(tokens)
                vocabulary.update(tokens)

    if class_counts["fake"] == 0 or class_counts["true"] == 0:
        return None

    return NaiveBayesModel(
        class_counts=class_counts,
        token_counts=token_counts,
        total_tokens=total_tokens,
        vocabulary=vocabulary,
    )


def _class_log_score(label: str, article_counts: Counter[str], model: NaiveBayesModel) -> float:
    class_total = sum(model.class_counts.values())
    log_score = math.log(model.class_counts[label] / class_total)
    vocabulary_size = len(model.vocabulary)
    denominator = model.total_tokens[label] + vocabulary_size

    for token, count in article_counts.items():
        if token not in model.vocabulary:
            continue
        numerator = model.token_counts[label][token] + 1
        log_score += count * math.log(numerator / denominator)
    return log_score


def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    exp_scores = {
        label: math.exp(score - max_score)
        for label, score in log_scores.items()
    }
    total = sum(exp_scores.values())
    return {
        label: value / total
        for label, value in exp_scores.items()
    }


def _top_indicators(
    article_counts: Counter[str],
    model: NaiveBayesModel,
    *,
    predicted_label: str,
    limit: int = 8,
) -> list[str]:
    scored_terms: list[tuple[float, str]] = []
    for token, count in article_counts.items():
        if token not in model.vocabulary:
            continue
        fake_log_prob = _token_log_probability("fake", token, model)
        true_log_prob = _token_log_probability("true", token, model)
        log_odds = fake_log_prob - true_log_prob
        direction_score = log_odds if predicted_label == "FAKE" else -log_odds
        scored_terms.append((direction_score * count, token))

    scored_terms.sort(reverse=True)
    return [token for score, token in scored_terms[:limit] if score > 0]


def _token_log_probability(label: str, token: str, model: NaiveBayesModel) -> float:
    vocabulary_size = len(model.vocabulary)
    denominator = model.total_tokens[label] + vocabulary_size
    numerator = model.token_counts[label][token] + 1
    return math.log(numerator / denominator)


def _tokenize(text: str) -> list[str]:
    return [
        word.lower().strip("'")
        for word in _WORD_PATTERN.findall(text)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    ]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
