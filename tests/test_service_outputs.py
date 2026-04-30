"""Tests for service-level CrewAI output handling."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from detector_fake_news.models import BiasReport, ClaimExtraction, EvidenceItem, FinalVerdict, LegalCase
from detector_fake_news.service import _extract_structured_outputs, _recover_claims_from_cases


@dataclass
class FakeTaskOutput:
    pydantic: object


class ServiceOutputTests(unittest.TestCase):
    def test_extract_structured_outputs_tolerates_async_task_order(self) -> None:
        claims = ClaimExtraction(claims=["A claim was made."])
        supporting = LegalCase(
            stance="supporting",
            case_summary="Support.",
            results=[
                EvidenceItem(
                    claim="A claim was made.",
                    verdict="SUPPORTED",
                    confidence=0.8,
                    reasoning="Supported by evidence.",
                    evidence="Evidence.",
                    source_urls=[],
                )
            ],
        )
        opposing = LegalCase(
            stance="opposing",
            case_summary="Opposition.",
            results=[
                EvidenceItem(
                    claim="A claim was made.",
                    verdict="UNVERIFIABLE",
                    confidence=0.6,
                    reasoning="No external source.",
                    evidence="No source.",
                    source_urls=[],
                )
            ],
        )
        bias = BiasReport(
            tone="Neutral",
            bias_score=0.2,
            flags=[],
            reasoning="Mostly factual tone.",
        )
        verdict = FinalVerdict(
            label="MIXED",
            truth_score=0.5,
            confidence=0.7,
            summary="Mixed evidence.",
            explanation="Some claims are unresolved.",
            claim_verdicts=["SUPPORTED", "UNVERIFIABLE"],
            bias_score=0.2,
            tone="Neutral",
        )

        extracted = _extract_structured_outputs(
            [
                FakeTaskOutput(supporting),
                FakeTaskOutput(opposing),
                FakeTaskOutput(claims),
                FakeTaskOutput(verdict),
                FakeTaskOutput(bias),
            ]
        )

        self.assertEqual(extracted, (claims, supporting, opposing, bias, verdict))

    def test_extract_structured_outputs_recovers_missing_claim_extraction(self) -> None:
        supporting = _case("supporting", "Recovered claim.")
        opposing = _case("opposing", "Recovered claim.")
        bias = BiasReport(
            tone="Neutral",
            bias_score=0.2,
            flags=[],
            reasoning="Mostly factual tone.",
        )
        verdict = FinalVerdict(
            label="UNVERIFIABLE",
            truth_score=0.5,
            confidence=0.5,
            summary="Not enough evidence.",
            explanation="The claim needs better sourcing.",
            claim_verdicts=["UNVERIFIABLE"],
            bias_score=0.2,
            tone="Neutral",
        )

        claims, _, _, _, _ = _extract_structured_outputs(
            [
                FakeTaskOutput(supporting),
                FakeTaskOutput(opposing),
                FakeTaskOutput(bias),
                FakeTaskOutput(verdict),
            ]
        )

        self.assertEqual(claims.claims, ["Recovered claim."])

    def test_recover_claims_from_cases_deduplicates_claims(self) -> None:
        claims = _recover_claims_from_cases(
            _case("supporting", "Same claim."),
            _case("opposing", "same claim."),
        )

        self.assertEqual(claims, ["Same claim."])


def _case(stance: str, claim: str) -> LegalCase:
    return LegalCase(
        stance=stance,
        case_summary=f"{stance} case.",
        results=[
            EvidenceItem(
                claim=claim,
                verdict="UNVERIFIABLE",
                confidence=0.6,
                reasoning="No external source.",
                evidence="No source.",
                source_urls=[],
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
