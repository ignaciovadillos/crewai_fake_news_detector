"""Tests for batch diagnostics helpers."""

from __future__ import annotations

import unittest

from detector_fake_news.diagnostics import batch_log_filename, describe_exception


class DiagnosticsTests(unittest.TestCase):
    def test_batch_log_filename_uses_run_id(self) -> None:
        self.assertEqual(
            batch_log_filename("batch-123"),
            "batch-123-diagnostics.csv",
        )

    def test_describe_exception_includes_type_and_message(self) -> None:
        try:
            raise ValueError("bad row")
        except ValueError as exc:
            details = describe_exception(exc)

        self.assertEqual(details["error_type"], "ValueError")
        self.assertEqual(details["error"], "bad row")
        self.assertIn("ValueError: bad row", details["traceback"])

    def test_describe_exception_handles_missing_exception(self) -> None:
        details = describe_exception(None)

        self.assertEqual(details["error_type"], "")
        self.assertEqual(details["error"], "")
        self.assertEqual(details["traceback"], "")


if __name__ == "__main__":
    unittest.main()
