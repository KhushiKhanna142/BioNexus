"""
Pydantic v2 schemas for the BioNexus v1 Product API.
Strict validation, OpenAPI examples, and RFC 7807 Problem Details.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

ConfidenceTier = Literal["high", "medium", "low"]
PAM50Subtype = Literal["LumA", "LumB", "Her2", "Basal", "Normal"]


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details for HTTP APIs."""
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str
    invalid_params: list[dict[str, Any]] | None = None


class ModalityQCItem(BaseModel):
    modality: str
    status: Literal["PASSED", "WARNING", "FAILED"]
    total_expected: int
    mapped_count: int
    coverage_pct: float
    missing_pct: float
    value_min: float
    value_max: float
    value_mean: float
    detected_scale: str = "unknown"
    likely_raw_counts: bool
    duplicate_ids_count: int = 0
    duplicate_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    fix_hints: list[str] = Field(default_factory=list)
    auto_normalized_applied: bool = False


class CaseQCResponse(BaseModel):
    overall_status: Literal["PASSED", "WARNING", "FAILED"]
    modalities_checked: list[str]
    can_proceed: bool
    requires_caution: bool = False
    ready_modalities_count: int = 0
    summary_message: str
    auto_normalized_modalities: list[str] = Field(default_factory=list)
    modality_details: dict[str, ModalityQCItem]



class CaseCreateResponse(BaseModel):
    case_id: str
    patient_ref: str
    specimen_type: str
    status: str
    created_at: str
    modalities_received: list[str]
    qc: CaseQCResponse


class CaseCreateJSONRequest(BaseModel):
    patient_ref: str = Field(description="Pseudonymous patient identifier (alphanumeric/hyphens only)", example="PATIENT-BRCA-042")
    specimen_type: str = Field(default="primary_tumor", example="primary_tumor")
    rna: dict[str, float] | None = Field(default=None, description="RNA-seq log2(TPM + 1) gene expression values")
    mirna: dict[str, float] | None = Field(default=None, description="miRNA log2(RPM + 1) quantification values")
    meth: dict[str, float] | None = Field(default=None, description="DNA methylation HM450 beta values [0, 1]")


class BiomarkerAttributionItem(BaseModel):
    gene: str = Field(description="Gene symbol or probe name")
    feature_id: str = Field(description="Internal feature identifier (Ensembl / probe ID)")
    attribution: float = Field(description="Signed Integrated Gradients attribution score")
    importance: float = Field(description="Magnitude of attribution score |score|")
    direction: Literal["pushes toward predicted class", "pushes away from predicted class"]
    layer: str


class NextTestRecommendationItem(BaseModel):
    assay: str = Field(example="miRNA Sequencing")
    expected_gain: float = Field(description="Estimated gain in diagnostic confidence / accuracy", example=0.082)
    cost: float = Field(description="Placeholder cost in USD", example=350.0)
    turnaround: str = Field(example="3-5 days")
    gain_per_cost: float = Field(description="Cost efficiency ratio", example=0.000234)
    rationale: str = Field(example="Adding miRNA sequencing provides highest subtype differentiation power.")


class CasePredictResponse(BaseModel):
    case_id: str
    patient_ref: str
    predicted_subtype: PAM50Subtype
    confidence: float = Field(ge=0.0, le=1.0)
    calibrated_probabilities: dict[PAM50Subtype, float]
    confidence_tier: ConfidenceTier
    abstain: bool
    abstain_reason: str | None = None
    modalities_used: list[str]
    modality_weights: dict[str, float]
    top_attributed_genes: dict[str, list[BiomarkerAttributionItem]]
    next_test_recommendations: list[NextTestRecommendationItem]
    model_version: str
    preprocessing_version: str
    qc_summary: dict[str, Any]
    timings_ms: dict[str, float]
    disclaimer: str = "Research use only, not a diagnostic device."


class CaseListResponse(BaseModel):
    total: int
    cases: list[dict[str, Any]]


class PanelPredictRequest(BaseModel):
    patient_ref: str = "PATIENT-PANEL-001"
    genes: dict[str, float] = Field(description="Dictionary of 10-50 gene symbols/Ensembl IDs and their log2 expression")


class PanelPredictResponse(BaseModel):
    patient_ref: str
    predicted_subtype: PAM50Subtype
    confidence: float
    probabilities: dict[PAM50Subtype, float]
    panel_size: int
    genes_matched: int
    model_type: str = "Lightweight Student MLP (Graph-Free)"
    disclaimer: str = "Research use only, not a diagnostic device."


class CostAccuracyCurvePoint(BaseModel):
    assays: list[str]
    estimated_cost: float
    accuracy: float
    macro_f1: float
    confidence: float


class CostAccuracyResponse(BaseModel):
    source: str = "precomputed_ablation_benchmarks"
    budget_evaluated: float | None = None
    recommended_path: list[CostAccuracyPoint] if False else list[Any]
    curve: list[dict[str, Any]]
