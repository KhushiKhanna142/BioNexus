"""Pydantic schemas for the BioNexus serving API.
Zero hard-coded values, fake defaults, or mock fixtures.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field

PAM50 = Literal["LumA", "LumB", "Her2", "Basal", "Normal"]
Layer = Literal["rna", "mirna", "meth"]


class ErrorBody(BaseModel):
    error: str
    detail: str | None = None
    missing: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    mode: Literal["live", "unloaded"]
    model_loaded: bool = False
    model_source: str | None = None
    model_version: str | None = None
    model_load_error: str | None = None
    model_uri: str | None = None
    missing_artifacts: list[str] = Field(default_factory=list)
    artifact_versions: dict[str, str] = Field(default_factory=dict)
    manifest_hash: str | None = None
    model_card: dict[str, Any] | None = None
    num_patients: int = 0
    allow_mock: bool = False


class PatientListItem(BaseModel):
    id: str
    true_pam50: str
    predicted_subtype: str | None = None
    confidence: float | None = None
    fold_id: int | None = None
    prediction_type: Literal["out_of_fold", "new_patient (experimental, unvalidated)"] | None = None
    rna_available: bool = True
    mirna_available: bool = True
    methylation_available: bool = True


class PredictionRequest(BaseModel):
    patient_id: str


class PredictionResponse(BaseModel):
    patient_id: str
    predicted_subtype: str
    confidence: float
    probabilities: dict[str, float]
    modality_weights: dict[str, float]
    true_pam50: str
    fold_id: int
    prediction_type: Literal["out_of_fold", "new_patient (experimental, unvalidated)"] = "out_of_fold"
    match: bool
    description: str = ""


class BiomarkerEntry(BaseModel):
    gene: str
    feature_id: str
    importance: float
    attribution: float
    effect: Literal["positive", "negative"]
    layer: str


class ExplanationResponse(BaseModel):
    patient_id: str
    layer: str
    biomarkers: list[BiomarkerEntry]
    method: str = "Integrated Gradients"
    steps: int = 50


class CohortResponse(BaseModel):
    total_patients: int
    subtype_distribution: dict[str, int]
    funnel: list[dict[str, Any]] | None = None
    cv_label: str = "5-fold cross-validation within TCGA-BRCA; no external validation"


class GraphNodeModel(BaseModel):
    id: str
    subtype: str
    degree: int
    index: int


class GraphEdgeModel(BaseModel):
    source: str
    target: str
    weight: float | None = None


class GraphResponse(BaseModel):
    layer: str
    nodes: list[GraphNodeModel]
    edges: list[GraphEdgeModel]
    stats: dict[str, Any]
    neighborhood_of: str | None = None
    hops: int | None = None
    layout_note: str = "Layout positions have no biological meaning."


# =========================================================================
# Patient Graph Explorer Schemas
# =========================================================================

class NeighborLabelPurity(BaseModel):
    overall: float
    per_subtype: dict[str, float]


class GraphLayerSummary(BaseModel):
    nodes: int
    edges: int
    mean_degree: float
    min_degree: int
    neighbor_label_purity: NeighborLabelPurity
    connected_components: int


class GraphSummaryResponse(BaseModel):
    rna: GraphLayerSummary
    mirna: GraphLayerSummary
    meth: GraphLayerSummary


class EgoNodeModel(BaseModel):
    patient_id: str
    pam50: str
    predicted_pam50: str | None = None
    confidence: float | None = None
    hop: int
    degree: int
    is_directed_neighbor: bool = False
    is_direct_knn: bool = False
    similarity_to_center: float | None = None
    fold_id: int | None = None


class EgoEdgeModel(BaseModel):
    source: str
    target: str
    cosine_similarity: float
    directed_knn_flag: bool = False
    is_direct_knn: bool = False


class EgoStatsModel(BaseModel):
    n_nodes: int
    n_edges: int
    subtype_counts: dict[str, int]
    same_subtype_fraction: float
    mean_similarity: float
    truncated: bool = False


class ClinicalDetailsModel(BaseModel):
    patient_id: str
    true_pam50: str
    predicted_pam50: str | None = None
    confidence: float | None = None
    age: str = "Not available"
    stage: str = "Not available"
    histology: str = "Not available"
    grade: str = "Not available"
    node_status: str = "Not available"
    er_status: str = "Not available"
    pr_status: str = "Not available"
    her2_status: str = "Not available"
    overall_survival_months: float | str = "Not available"
    overall_survival_status: str = "Not available"
    cancer_type: str = "Breast Invasive Carcinoma (TCGA-BRCA)"


class TopNeighborItem(BaseModel):
    rank: int
    patient_id: str
    pam50: str
    similarity: float
    is_directed: bool = True


class CrossModalityNeighborItem(BaseModel):
    patient_id: str
    rna: float
    mirna: float
    methylation: float
    shared: bool = False


class GraphStatsDetailsModel(BaseModel):
    total_patients: int
    total_edges: int
    neighbors_k: int
    mean_similarity: float
    same_subtype_neighbors: str


class PatientEgoGraphResponse(BaseModel):
    patient_id: str
    layer: str
    hops: int
    center_node: EgoNodeModel
    nodes: list[EgoNodeModel]
    edges: list[EgoEdgeModel]
    stats: EgoStatsModel
    top_10_neighbors: list[TopNeighborItem] = Field(default_factory=list)
    cross_modality_neighbors: list[CrossModalityNeighborItem] = Field(default_factory=list)
    clinical_details: ClinicalDetailsModel | None = None
    graph_stats_details: GraphStatsDetailsModel | None = None


class LayerNeighborCompare(BaseModel):
    neighbors: list[str]
    same_subtype_fraction: float
    mean_similarity: float
    count: int


class PatientGraphCompareResponse(BaseModel):
    patient_id: str
    true_pam50: str
    layers: dict[str, LayerNeighborCompare]
    overlap_counts: dict[str, int]
    consensus_neighbors: list[str]
    all_neighbors: list[str]


class GraphOverviewResponse(BaseModel):
    layer: str
    nodes: list[EgoNodeModel]
    edges: list[EgoEdgeModel]
    stats: dict[str, Any]
    thinned_for_display: bool = True
    layout_note: str = "layout only, positions have no biological meaning"


class ProfileFeature(BaseModel):
    feature_id: str
    symbol: str
    value: float
    z_score: float | None = None


class ProfileResponse(BaseModel):
    patient_id: str
    layer: str
    n_total: int
    features: list[ProfileFeature]


# =========================================================================
# Additional Real Schemas (Phase 2 endpoints)
# =========================================================================

class AblationRequest(BaseModel):
    patient_id: str
    layer: Literal["rna", "mirna", "meth"] = "rna"
    genes: list[str] = Field(default_factory=list)
    mode: Literal["custom", "top_k", "random_k"] = "custom"
    k: int = 20
    seed: int = 42


class AblationResponse(BaseModel):
    patient_id: str
    layer: str
    mode: str
    ablated_features: list[str]
    target_class: str
    original_logit: float
    ablated_logit: float
    logit_change: float
    original_probabilities: dict[str, float]
    ablated_probabilities: dict[str, float]
    probability_change: dict[str, float]


class HeatmapResponse(BaseModel):
    layer: str
    patient_ids: list[str]
    feature_ids: list[str]
    symbols: list[str]
    z_scores: list[list[float]]


class EmbeddingPoint(BaseModel):
    patient_id: str
    true_pam50: str
    predicted_pam50: str | None = None
    x: float
    y: float


class EmbeddingsResponse(BaseModel):
    space: str
    method: str
    points: list[EmbeddingPoint]
    explained_variance: float | None = None


class SurvivalResponse(BaseModel):
    group_by: str
    subtypes: dict[str, Any]
    logrank_p_luma_vs_basal: float | None = None


class AttentionResponse(BaseModel):
    patient_id: str
    layer_names: list[str]
    cross_attention_matrix: list[list[float]]
    pooling_weights: dict[str, float]
    label_note: str = (
        "Multi-modal fusion layer cross-attention between RNA, miRNA, and Methylation representations. "
        "This is NOT GraphGPS graph global attention."
    )


class PredictNewResponse(BaseModel):
    prediction_type: str = "new_patient (experimental, unvalidated)"
    predicted_subtype: str
    confidence: float
    probabilities: dict[str, float]
    fold_probabilities: dict[str, dict[str, float]]
    modality_weights: dict[str, float]
    nearest_neighbors: dict[str, list[dict[str, Any]]]
    warning: str = (
        "Experimental inductive prediction. The model was trained transductively on TCGA-BRCA (646 patients). "
        "Predictions on external uploaded data are not clinically validated."
    )
