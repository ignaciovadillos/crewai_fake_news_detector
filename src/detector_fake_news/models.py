"""Pydantic models for structured crew outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ClaimVerdict = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE", "MIXED"]
ArticleLabel = Literal["REAL", "FAKE", "MIXED", "UNVERIFIABLE"]


class ClaimExtraction(BaseModel):
    claims: list[str] = Field(
        ...,
        description="Short list of factual claims extracted from the article.",
        min_length=1,
        max_length=5,
    )


class EvidenceItem(BaseModel):
    claim: str
    verdict: ClaimVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    evidence: str
    source_urls: list[str] = Field(default_factory=list)


class LegalCase(BaseModel):
    stance: Literal["supporting", "opposing"]
    case_summary: str
    results: list[EvidenceItem]


class BiasReport(BaseModel):
    tone: str
    bias_score: float = Field(..., ge=0.0, le=1.0)
    flags: list[str]
    reasoning: str


class FinalVerdict(BaseModel):
    label: ArticleLabel
    truth_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    explanation: str
    claim_verdicts: list[str]
    bias_score: float = Field(..., ge=0.0, le=1.0)
    tone: str


class BaselinePrediction(BaseModel):
    label: Literal["REAL", "FAKE"]
    real_probability: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_indicators: list[str] = Field(default_factory=list)
    training_examples: int = 0


class EvidenceQualityReport(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    grade: Literal["LOW", "MEDIUM", "HIGH"]
    source_count: int = 0
    online_source_count: int = 0
    offline_source_count: int = 0
    unsupported_claim_count: int = 0
    notes: list[str] = Field(default_factory=list)


class ContradictionReport(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    level: Literal["LOW", "MEDIUM", "HIGH"]
    contradiction_count: int = 0
    mixed_signal_count: int = 0
    notes: list[str] = Field(default_factory=list)


class PipelineReport(BaseModel):
    title: str = ""
    article_text: str
    claims: ClaimExtraction
    supporting_case: LegalCase
    opposing_case: LegalCase
    bias_report: BiasReport
    final_verdict: FinalVerdict
    baseline_prediction: BaselinePrediction | None = None
    evidence_quality: EvidenceQualityReport | None = None
    contradiction_report: ContradictionReport | None = None
    raw_output: str = ""
