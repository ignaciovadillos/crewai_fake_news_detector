"""Tests for deterministic helper functions."""

from __future__ import annotations

import unittest

from detector_fake_news.article_fetcher import _normalize_url
from detector_fake_news.contradictions import detect_contradictions
from detector_fake_news.evidence_quality import score_evidence_quality
from detector_fake_news.models import EvidenceItem, LegalCase
from detector_fake_news.reporting import report_filename


class HelperTests(unittest.TestCase):
    def test_normalize_url_adds_https(self) -> None:
        self.assertEqual(
            _normalize_url("example.com/article"),
            "https://example.com/article",
        )

    def test_normalize_url_rejects_invalid_scheme(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_url("ftp://example.com/article")

    def test_report_filename_is_safe_markdown_name(self) -> None:
        self.assertEqual(
            report_filename("A Test: Article / Demo?"),
            "a-test-article-demo-report.md",
        )

    def test_evidence_quality_scores_high_for_trusted_cited_source(self) -> None:
        case = LegalCase(
            stance="supporting",
            case_summary="Strong evidence.",
            results=[
                EvidenceItem(
                    claim="The claim was verified.",
                    verdict="SUPPORTED",
                    confidence=0.9,
                    reasoning="Reuters corroborates the claim.",
                    evidence="Reuters report.",
                    source_urls=["https://www.reuters.com/world/example"],
                )
            ],
        )

        quality = score_evidence_quality(case)

        self.assertEqual(quality.grade, "HIGH")
        self.assertEqual(quality.online_source_count, 1)
        self.assertEqual(quality.unsupported_claim_count, 0)

    def test_evidence_quality_scores_low_without_sources(self) -> None:
        case = LegalCase(
            stance="supporting",
            case_summary="Weak evidence.",
            results=[
                EvidenceItem(
                    claim="The claim was verified.",
                    verdict="SUPPORTED",
                    confidence=0.9,
                    reasoning="The article says it.",
                    evidence="Article text only.",
                    source_urls=[],
                )
            ],
        )

        quality = score_evidence_quality(case)

        self.assertEqual(quality.grade, "LOW")
        self.assertEqual(quality.source_count, 0)
        self.assertEqual(quality.unsupported_claim_count, 1)

    def test_detect_contradictions_flags_strong_conflict(self) -> None:
        supporting = LegalCase(
            stance="supporting",
            case_summary="Support.",
            results=[
                EvidenceItem(
                    claim="Water contamination was confirmed by a federal agency.",
                    verdict="SUPPORTED",
                    confidence=0.9,
                    reasoning="External source confirms it.",
                    evidence="Source A.",
                    source_urls=["https://www.reuters.com/example"],
                )
            ],
        )
        opposing = LegalCase(
            stance="opposing",
            case_summary="Opposition.",
            results=[
                EvidenceItem(
                    claim="Water contamination was not confirmed by a federal agency.",
                    verdict="CONTRADICTED",
                    confidence=0.9,
                    reasoning="External source rejects it.",
                    evidence="Source B.",
                    source_urls=["https://apnews.com/example"],
                )
            ],
        )

        report = detect_contradictions(supporting, opposing)

        self.assertEqual(report.level, "HIGH")
        self.assertEqual(report.contradiction_count, 1)


if __name__ == "__main__":
    unittest.main()
