"""Tests for lightweight UI helper logic."""

from __future__ import annotations

import unittest

from detector_fake_news.ui_helpers import (
    baseline_agreement,
    expected_match,
    normalize_expected_label,
    rows_to_csv,
)


class UIHelperTests(unittest.TestCase):
    def test_baseline_agreement_matches_decisive_labels(self) -> None:
        self.assertEqual(baseline_agreement("REAL", "REAL"), "AGREE")
        self.assertEqual(baseline_agreement("FAKE", "REAL"), "DISAGREE")

    def test_baseline_agreement_ignores_non_binary_agent_labels(self) -> None:
        self.assertEqual(baseline_agreement("MIXED", "REAL"), "N/A")
        self.assertEqual(baseline_agreement("UNVERIFIABLE", "FAKE"), "N/A")

    def test_expected_label_normalization(self) -> None:
        self.assertEqual(normalize_expected_label("true"), "REAL")
        self.assertEqual(normalize_expected_label("false"), "FAKE")
        self.assertEqual(normalize_expected_label("mixed"), "MIXED")

    def test_expected_match(self) -> None:
        self.assertEqual(expected_match("true", "REAL"), "CORRECT")
        self.assertEqual(expected_match("fake", "REAL"), "INCORRECT")
        self.assertEqual(expected_match("", "REAL"), "NO EXPECTED LABEL")
        self.assertEqual(expected_match("true", "ERROR"), "NOT SCORED")

    def test_rows_to_csv_preserves_keys_from_later_rows(self) -> None:
        csv_text = rows_to_csv(
            [
                {"title": "First"},
                {"title": "Second", "extra": "value"},
            ]
        )

        self.assertIn("extra", csv_text.splitlines()[0])
        self.assertIn("value", csv_text)


if __name__ == "__main__":
    unittest.main()
