// ============== Core Types ==============

export type PAM50Subtype = 'LumA' | 'LumB' | 'Her2' | 'Basal' | 'Normal';

export interface Patient {
  id: string;
  subtype?: PAM50Subtype;
  true_pam50?: PAM50Subtype;
  predicted_subtype?: PAM50Subtype;
  confidence?: number;
  fold_id?: number;
  prediction_type?: string;
  rna_available: boolean;
  mirna_available: boolean;
  methylation_available: boolean;
  clinical?: ClinicalData;
}

export interface ClinicalData {
  age?: number;
  er_status?: string;
  tumor_grade?: number;
  lymph_node_status?: string;
}

export interface PredictionResult {
  patient_id: string;
  predicted_subtype: PAM50Subtype;
  confidence: number;
  probabilities: SubtypeProbabilities;
  modality_weights: ModalityWeights;
  description: string;
  true_pam50?: PAM50Subtype;
  fold_id?: number;
  prediction_type?: string;
  match?: boolean;
}

export interface SubtypeProbabilities {
  LumA: number;
  LumB: number;
  Her2: number;
  Basal: number;
  Normal: number;
}

export interface ModalityWeights {
  rna: number;
  mirna: number;
  methylation: number;
}

export interface BiomarkerEntry {
  gene: string;
  feature_id?: string;
  importance: number;
  attribution?: number;
  effect: 'positive' | 'negative';
  layer?: string;
}

// ============== Cost vs Accuracy Simulation Types ==============

export interface PolicyPoint {
  threshold: number;
  avg_cost: number;
  macro_f1: number;
  macro_f1_ci_lower: number;
  macro_f1_ci_upper: number;
  accuracy: number;
  accuracy_ci_lower: number;
  accuracy_ci_upper: number;
  escalation_rate?: number;
  policy_type?: string;
  pct_of_full_f1?: number;
  pct_of_full_accuracy?: number;
  cost_savings_pct?: number;
}

export interface BaselineCurve {
  name: string;
  cost: number;
  macro_f1: number;
  macro_f1_ci?: [number, number];
  accuracy: number;
  accuracy_ci?: [number, number];
  description?: string;
}

export interface AssayCostItem {
  assay: string;
  full_name: string;
  unit_cost: number;
  is_placeholder: boolean;
  badge: string;
  turnaround: string;
  features_count: number;
  standalone_f1: number;
  role: string;
}

export interface ModalityCombinationItem {
  id: string;
  name: string;
  modalities: string[];
  cost: number;
  macro_f1: number;
  macro_f1_std: number;
  macro_f1_ci_lower?: number;
  macro_f1_ci_upper?: number;
  accuracy: number;
  accuracy_std: number;
  accuracy_ci_lower?: number;
  accuracy_ci_upper?: number;
  num_features?: number;
}

export interface CostAccuracyResponse {
  status: string;
  empty: boolean;
  message?: string;
  headline: string;
  caveat: string;
  is_placeholder: boolean;
  placeholder_badge: string;
  readout: string;
  nearest_point: PolicyPoint | null;
  policy_points: PolicyPoint[];
  baselines: {
    always_all?: BaselineCurve;
    never_add?: BaselineCurve;
  };
  gain_table: {
    is_placeholder?: boolean;
    placeholder_badge?: string;
    caveat?: string;
    headline?: string;
    assays?: AssayCostItem[];
  };
  modality_combinations: ModalityCombinationItem[];
  curve?: Array<{
    assays: string[];
    estimated_cost: number;
    accuracy: number;
    macro_f1: number;
    mean_confidence?: number;
  }>;
}

// ============== QC & Case Intake Types ==============

export interface QCModalityItem {
  modality: string;
  status: 'PASSED' | 'WARNING' | 'FAILED';
  total_expected: number;
  mapped_count: number;
  coverage_pct: number;
  missing_pct: number;
  value_min: number;
  value_max: number;
  value_mean: number;
  detected_scale: string;
  likely_raw_counts: boolean;
  duplicate_ids_count: number;
  duplicate_ids: string[];
  warnings: string[];
  errors: string[];
  fix_hints: string[];
  auto_normalized_applied: boolean;
}

export interface CaseQCReportResponse {
  overall_status: 'PASSED' | 'WARNING' | 'FAILED';
  modalities_checked: string[];
  can_proceed: boolean;
  requires_caution: boolean;
  ready_modalities_count: number;
  summary_message: string;
  auto_normalized_modalities: string[];
  results: Record<string, QCModalityItem>;
}

// ============== Batch Triage Types ==============

export interface BatchCardItem {
  case_id: string;
  patient_ref: string;
  predicted_subtype: string;
  confidence: number;
  confidence_tier: string;
  triage_lane: 'auto_report' | 'clinician_review' | 'needs_more_data';
  data_types_used: string[];
  estimated_cost: number;
  status: string;
  abstain?: boolean;
  demo_data: boolean;
}

export interface BatchSummary {
  total_cases: number;
  auto_reported_count: number;
  auto_reported_pct: number;
  clinician_review_count: number;
  clinician_review_pct: number;
  needs_more_data_count: number;
  needs_more_data_pct: number;
  average_cost_per_case: number;
}

export interface BatchResponse {
  batch_id: string;
  status: string;
  total_cases: number;
  summary?: BatchSummary;
  lanes?: {
    auto_report: BatchCardItem[];
    clinician_review: BatchCardItem[];
    needs_more_data: BatchCardItem[];
  };
  results?: BatchCardItem[];
}



export interface ExplanationResult {
  patient_id: string;
  biomarkers: BiomarkerEntry[];
  method: string;
  layer?: string;
  steps?: number;
}

export interface ModelMetrics {
  accuracy: number;
  macro_f1: number;
  per_class: Record<PAM50Subtype, { precision: number; recall: number; f1: number }>;
  confusion_matrix: number[][];
  single_omics: {
    rna: { accuracy: number; f1: number; std: number };
    mirna: { accuracy: number; f1: number; std: number };
    methylation: { accuracy: number; f1: number; std: number };
  };
  fused: { accuracy: number; f1: number; std: number };
}

export interface CohortStats {
  total_patients: number;
  subtype_distribution: Record<string, number>;
  funnel: FunnelStep[];
  cv_label?: string;
}

export interface FunnelStep {
  label: string;
  count: number;
  description: string;
}

export interface GraphNode {
  id: string;
  subtype: PAM50Subtype;
  degree?: number;
  index?: number;
  x?: number;
  y?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number;
}

export interface PatientGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  layer?: string;
  stats?: Record<string, number>;
  layout_note?: string;
}

export interface OmicsProfile {
  patient_id: string;
  layer?: string;
  n_total?: number;
  features: { feature_id: string; symbol: string; value: number }[];
}

export interface PathwayEntry {
  name: string;
  p_value: number;
  neg_log10_p: number;
}

export interface CausalValidation {
  attributed_impact_median: number;
  random_impact_median: number;
  patients_larger_impact: number;
  total_patients: number;
  p_value: number;
  fold_difference: number;
  precomputed?: boolean;
  wording?: string;
}

// ============== Graph Explorer Types ==============

export interface NeighborLabelPurity {
  overall: number;
  per_subtype: Record<string, number>;
}

export interface GraphLayerSummary {
  nodes: number;
  edges: number;
  mean_degree: number;
  min_degree: number;
  neighbor_label_purity: NeighborLabelPurity;
  connected_components: number;
}

export interface GraphSummaryResponse {
  rna: GraphLayerSummary;
  mirna: GraphLayerSummary;
  meth: GraphLayerSummary;
}

export interface EgoNodeModel {
  patient_id: string;
  pam50: PAM50Subtype;
  predicted_pam50?: PAM50Subtype;
  hop: number;
  degree: number;
  is_directed_neighbor: boolean;
  fold_id?: number;
}

export interface EgoEdgeModel {
  source: string;
  target: string;
  cosine_similarity: number;
  directed_knn_flag: boolean;
}

export interface EgoStatsModel {
  n_nodes: number;
  n_edges: number;
  subtype_counts: Record<string, number>;
  same_subtype_fraction: number;
  mean_similarity: number;
  truncated?: boolean;
}

export interface ClinicalDetailsModel {
  patient_id: string;
  true_pam50: string;
  predicted_pam50?: string;
  confidence?: number;
  age: number | string;
  er_status: string;
  pr_status: string;
  her2_status: string;
  tumor_grade: string;
  lymph_node_status: string;
  histology: string;
  cancer_type: string;
}

export interface TopNeighborItem {
  rank: number;
  patient_id: string;
  pam50: PAM50Subtype;
  similarity: number;
  is_directed: boolean;
}

export interface CrossModalityNeighborItem {
  patient_id: string;
  rna: number;
  mirna: number;
  methylation: number;
  shared: boolean;
}

export interface GraphStatsDetailsModel {
  total_patients: number;
  total_edges: number;
  neighbors_k: number;
  mean_similarity: number;
  same_subtype_neighbors: string;
}

export interface PatientEgoGraphResponse {
  patient_id: string;
  layer: string;
  hops: number;
  center_node: EgoNodeModel;
  nodes: EgoNodeModel[];
  edges: EgoEdgeModel[];
  stats: EgoStatsModel;
  top_10_neighbors?: TopNeighborItem[];
  cross_modality_neighbors?: CrossModalityNeighborItem[];
  clinical_details?: ClinicalDetailsModel;
  graph_stats_details?: GraphStatsDetailsModel;
}

export interface LayerNeighborCompare {
  neighbors: string[];
  same_subtype_fraction: number;
  mean_similarity: number;
  count: number;
}

export interface PatientGraphCompareResponse {
  patient_id: string;
  true_pam50: string;
  layers: Record<string, LayerNeighborCompare>;
  overlap_counts: {
    rna_mirna: number;
    rna_meth: number;
    mirna_meth: number;
    all_three: number;
  };
  consensus_neighbors: string[];
  all_neighbors: string[];
}

export interface GraphOverviewResponse {
  layer: string;
  nodes: EgoNodeModel[];
  edges: EgoEdgeModel[];
  stats: Record<string, any>;
  thinned_for_display: boolean;
  layout_note: string;
}

// ============== Subtype Metadata ==============

export const SUBTYPE_COLORS: Record<PAM50Subtype, string> = {
  LumA: '#4F6FE8',
  LumB: '#F47C9C',
  Her2: '#F5B942',
  Basal: '#8B6FE0',
  Normal: '#34C38F',
};

export const SUBTYPE_LABELS: Record<PAM50Subtype, string> = {
  LumA: 'Luminal A',
  LumB: 'Luminal B',
  Her2: 'HER2-enriched',
  Basal: 'Basal-like',
  Normal: 'Normal-like',
};

export const SUBTYPE_DESCRIPTIONS: Record<PAM50Subtype, string> = {
  LumA: 'Luminal A subtype is typically associated with hormone receptor positivity and better prognosis.',
  LumB: 'Luminal B subtype is hormone receptor positive but tends to grow faster than Luminal A.',
  Her2: 'HER2-enriched subtype overexpresses the HER2/neu protein and may benefit from targeted therapy.',
  Basal: 'Basal-like subtype lacks hormone receptors and HER2, often associated with more aggressive disease.',
  Normal: 'Normal-like subtype resembles normal breast tissue gene expression patterns.',
};

export const MODALITY_COLORS = {
  rna: '#8B6FE0',
  mirna: '#F5A623',
  methylation: '#2BB5A0',
};

export const MODALITY_LABELS = {
  rna: 'RNA-seq',
  mirna: 'miRNA',
  methylation: 'Methylation',
};

export const SUBTYPES: PAM50Subtype[] = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal'];

// ============== Blockchain & Security Types ==============

export interface BlockchainVerification {
  is_anchored: boolean;
  is_unaltered: boolean;
  status: 'VERIFIED_AUTHENTIC' | 'TAMPER_DETECTED' | 'NOT_YET_ANCHORED' | 'REVOKED_BY_PATIENT';
  anchored_fingerprint?: string;
  current_fingerprint?: string;
  timestamp?: number;
  tx_hash?: string;
  block_number?: number;
  authority?: string;
  polygonscan_url?: string;
  network?: string;
  model_version?: string;
  predicted_subtype?: number;
  confidence_bps?: number;
  message?: string;
}

export interface MerkleProof {
  has_proof: boolean;
  patient_id: string;
  leaf_hash: string;
  merkle_root: string;
  proof_path: string[];
  proof_length: number;
  is_valid: boolean;
  receipt?: Record<string, any>;
  error?: string;
}

export interface CohortBatchStatus {
  is_batch_anchored: boolean;
  merkle_root?: string;
  total_patients?: number;
  tree_depth?: number;
  timestamp?: number;
  on_chain_receipt?: {
    batch_identifier?: string;
    cohort_name?: string;
    merkle_root?: string;
    total_records?: number;
    timestamp?: number;
    authority?: string;
    tx_hash?: string;
    block_number?: number;
    network?: string;
    is_live_chain?: boolean;
    polygonscan_url?: string;
    method?: string;
  };
  wallet_address?: string;
  wallet_balance_pol?: string;
  contract_address?: string;
  message?: string;
}

export interface PatientSecurityStatus {
  patient_id: string;
  patient_hash: string;
  current_fingerprint: string;
  leaf_hash?: string;
  is_tampered: boolean;
  is_simulated_tamper: boolean;
  verification: BlockchainVerification;
  merkle_proof?: MerkleProof;
  cohort_status?: CohortBatchStatus;
}

