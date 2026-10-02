import type {
  Patient, PredictionResult, ExplanationResult, ModelMetrics,
  CohortStats, PatientGraph, OmicsProfile, PAM50Subtype, CausalValidation,
  GraphSummaryResponse, PatientEgoGraphResponse, PatientGraphCompareResponse, GraphOverviewResponse,
  CostAccuracyResponse, CaseQCReportResponse, BatchResponse,
  PatientSecurityStatus, CohortBatchStatus, MerkleProof,
} from '@/lib/types';
const BASE_URL = import.meta.env.VITE_API_URL ?? '';

export class ApiError extends Error {
  status: number;
  body: string;
  constructor(status: number, statusText: string, body: string) {
    super(`API error ${status}: ${statusText} - ${body}`);
    this.status = status;
    this.body = body;
  }
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit, signal?: AbortSignal): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const res = await fetch(url, {
    ...options,
    signal,
    headers: {
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': 'true',
      ...(options?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(res.status, res.statusText, body);
  }
  return res.json();
}

function mapPatient(raw: Record<string, unknown>): Patient {
  return {
    id: String(raw.id || raw.patient_id),
    subtype: (raw.true_pam50 || raw.subtype) as Patient['subtype'],
    true_pam50: raw.true_pam50 as Patient['true_pam50'],
    predicted_subtype: raw.predicted_subtype as Patient['predicted_subtype'],
    confidence: raw.confidence as number | undefined,
    fold_id: raw.fold_id as number | undefined,
    prediction_type: raw.prediction_type as string | undefined,
    rna_available: (raw.rna_available as boolean) ?? true,
    mirna_available: (raw.mirna_available as boolean) ?? true,
    methylation_available: (raw.methylation_available as boolean) ?? true,
  };
}

export const api = {
  async getPatients(signal?: AbortSignal): Promise<Patient[]> {
    const raw = await fetchAPI<Record<string, unknown>[]>('/api/patients', undefined, signal);
    return raw.map(mapPatient);
  },

  async predict(patientId: string, signal?: AbortSignal): Promise<PredictionResult> {
    const raw = await fetchAPI<Record<string, unknown>>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ patient_id: patientId }),
    }, signal);
    return {
      patient_id: String(raw.patient_id),
      predicted_subtype: raw.predicted_subtype as PAM50Subtype,
      confidence: Number(raw.confidence),
      probabilities: raw.probabilities as PredictionResult['probabilities'],
      modality_weights: raw.modality_weights as PredictionResult['modality_weights'],
      description: String(raw.description || ''),
      true_pam50: raw.true_pam50 as PAM50Subtype,
      fold_id: Number(raw.fold_id),
      prediction_type: String(raw.prediction_type || 'out_of_fold'),
      match: Boolean(raw.match),
    };
  },

  async getExplanation(patientId: string, topK = 20, layer = 'rna', signal?: AbortSignal): Promise<ExplanationResult> {
    const raw = await fetchAPI<Record<string, unknown>>(
      `/api/explain/${encodeURIComponent(patientId)}?top_k=${topK}&layer=${encodeURIComponent(layer)}`,
      undefined,
      signal,
    );
    const biomarkers = (raw.biomarkers as Record<string, unknown>[]).map((b) => ({
      gene: String(b.gene),
      feature_id: String(b.feature_id || b.gene),
      importance: Number(b.importance ?? Math.abs(Number(b.attribution ?? 0))),
      attribution: Number(b.attribution ?? b.importance ?? 0),
      effect: (b.effect as 'positive' | 'negative') || (Number(b.attribution) >= 0 ? 'positive' : 'negative'),
      layer: String(b.layer || raw.layer || layer),
    }));
    return {
      patient_id: String(raw.patient_id),
      biomarkers,
      method: String(raw.method || 'Integrated Gradients'),
      layer: String(raw.layer || layer),
      steps: Number(raw.steps || 50),
    };
  },

  async getMetrics(signal?: AbortSignal): Promise<ModelMetrics> {
    return fetchAPI('/api/metrics', undefined, signal);
  },

  async getCausalValidation(signal?: AbortSignal): Promise<CausalValidation> {
    const raw = await fetchAPI<Record<string, unknown>>('/api/causal', undefined, signal);
    return {
      attributed_impact_median: Number(raw.median_top_k_impact),
      random_impact_median: Number(raw.median_random_impact),
      patients_larger_impact: Number(raw.patients_top_k_beats_random),
      total_patients: Number(raw.n_patients),
      p_value: Number(raw.wilcoxon_p_value),
      fold_difference: Number(raw.fold_difference_median),
      precomputed: true,
      wording: String(raw.wording || ''),
    };
  },

  async getFusionDiagnosis(signal?: AbortSignal): Promise<{ observation?: string; mean_rna_weight_correct?: number; mean_rna_weight_incorrect?: number }> {
    return fetchAPI('/api/fusion-diagnosis', undefined, signal);
  },

  async getCohort(signal?: AbortSignal): Promise<CohortStats> {
    const raw = await fetchAPI<Record<string, unknown>>('/api/cohort', undefined, signal);
    return {
      total_patients: Number(raw.total_patients),
      subtype_distribution: raw.subtype_distribution as CohortStats['subtype_distribution'],
      funnel: (raw.funnel as CohortStats['funnel']) || [],
      cv_label: String(raw.cv_label || ''),
    };
  },

  async getGraphSummary(signal?: AbortSignal): Promise<GraphSummaryResponse> {
    return fetchAPI('/api/graphs/summary', undefined, signal);
  },

  async getPatientEgoGraph(
    layer = 'rna',
    patientId: string,
    hops = 1,
    signal?: AbortSignal
  ): Promise<PatientEgoGraphResponse> {
    const params = new URLSearchParams({ hops: String(hops) });
    return fetchAPI(
      `/api/graphs/${encodeURIComponent(layer)}/patient/${encodeURIComponent(patientId)}?${params}`,
      undefined,
      signal
    );
  },

  async getPatientGraphCompare(patientId: string, signal?: AbortSignal): Promise<PatientGraphCompareResponse> {
    return fetchAPI(`/api/graphs/patient/${encodeURIComponent(patientId)}/compare`, undefined, signal);
  },

  async getGraphOverview(layer = 'rna', maxNodes = 250, signal?: AbortSignal): Promise<GraphOverviewResponse> {
    const params = new URLSearchParams({ max_nodes: String(maxNodes) });
    return fetchAPI(`/api/graphs/${encodeURIComponent(layer)}/overview?${params}`, undefined, signal);
  },

  async getGraph(layer = 'rna', patientId?: string, hops = 1, signal?: AbortSignal): Promise<PatientGraph> {
    const params = new URLSearchParams({ layer });
    if (patientId) {
      params.set('patient_id', patientId);
      params.set('hops', String(hops));
    }
    return fetchAPI(`/api/graph?${params}`, undefined, signal);
  },

  async getProfile(patientId: string, layer = 'rna', topN = 30, signal?: AbortSignal): Promise<OmicsProfile> {
    const raw = await fetchAPI<Record<string, unknown>>(
      `/api/profile/${encodeURIComponent(patientId)}?layer=${encodeURIComponent(layer)}&top_n=${topN}`,
      undefined,
      signal,
    );
    const features = (raw.features as { feature_id: string; symbol: string; value: number }[]) || [];
    return {
      patient_id: String(raw.patient_id),
      layer: String(raw.layer),
      n_total: Number(raw.n_total || features.length),
      features: features.map((f) => ({
        feature_id: f.feature_id,
        symbol: f.symbol,
        value: f.value,
      })),
    };
  },

  async getPatientClinical(patientId: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
    return fetchAPI(`/api/patients/${encodeURIComponent(patientId)}`, undefined, signal);
  },

  async getHeatmap(layer = 'rna', patientIds?: string[], features?: string[], topN = 20, signal?: AbortSignal): Promise<{
    layer: string;
    patient_ids: string[];
    feature_ids: string[];
    symbols: string[];
    z_scores: number[][];
  }> {
    const params = new URLSearchParams({ layer, top_n: String(topN) });
    if (patientIds && patientIds.length) params.set('patient_ids', patientIds.join(','));
    if (features && features.length) params.set('features', features.join(','));
    return fetchAPI(`/api/heatmap?${params}`, undefined, signal);
  },

  async getEmbeddings(space = 'fused', method = 'pca', signal?: AbortSignal): Promise<{
    space: string;
    method: string;
    points: { patient_id: string; true_pam50: string; predicted_pam50?: string; x: number; y: number }[];
    explained_variance?: number;
  }> {
    return fetchAPI(`/api/embeddings?space=${encodeURIComponent(space)}&method=${encodeURIComponent(method)}`, undefined, signal);
  },

  async getSurvival(group = 'true', signal?: AbortSignal): Promise<{
    group_by: string;
    subtypes: Record<string, { timeline: number[]; survival_probability: number[]; median_survival: number | null; n_patients: number; n_events: number }>;
    logrank_p_luma_vs_basal?: number;
  }> {
    return fetchAPI(`/api/survival?group=${encodeURIComponent(group)}`, undefined, signal);
  },

  async getAttention(patientId: string, signal?: AbortSignal): Promise<{
    patient_id: string;
    layer_names: string[];
    cross_attention_matrix: number[][];
    pooling_weights: Record<string, number>;
    label_note: string;
  }> {
    return fetchAPI(`/api/attention/${encodeURIComponent(patientId)}`, undefined, signal);
  },

  async ablateGenes(payload: { patient_id: string; layer: string; genes?: string[]; mode?: 'custom' | 'top_k' | 'random_k'; k?: number; seed?: number }, signal?: AbortSignal): Promise<Record<string, unknown>> {
    return fetchAPI('/api/ablate', {
      method: 'POST',
      body: JSON.stringify(payload),
    }, signal);
  },

  async predictNewPatient(formData: FormData, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const url = `${BASE_URL}/api/predict-new`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(res.status, res.statusText, body);
    }
    return res.json();
  },

  async getPathways(): Promise<{ pathways: unknown[]; planned?: boolean; illustrative?: boolean; note?: string }> {
    return fetchAPI('/api/pathways');
  },

  async healthCheck(): Promise<{
    status: string;
    mode: string;
    num_patients: number;
    missing_artifacts?: string[];
    model_card?: Record<string, unknown>;
    allow_mock?: boolean;
  }> {
    return fetchAPI('/api/health');
  },

  async getModelInfo(): Promise<Record<string, unknown>> {
    return fetchAPI('/api/model-info');
  },

  async getCostAccuracy(
    budget?: number,
    threshold?: number,
    metric: 'macro_f1' | 'accuracy' = 'macro_f1',
    signal?: AbortSignal,
  ): Promise<CostAccuracyResponse> {
    const params = new URLSearchParams();
    if (budget !== undefined) params.append('budget', String(budget));
    if (threshold !== undefined) params.append('threshold', String(threshold));
    if (metric) params.append('metric', metric);
    const qs = params.toString() ? `?${params.toString()}` : '';
    return fetchAPI<CostAccuracyResponse>(`/api/simulate/cost-accuracy${qs}`, undefined, signal);
  },

  async runQC(formData: FormData, signal?: AbortSignal): Promise<CaseQCReportResponse> {
    const url = `${BASE_URL}/v1/qc`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(res.status, res.statusText, body);
    }
    return res.json();
  },

  async predictLive(formData: FormData, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const url = `${BASE_URL}/v1/predict-live`;
    const res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(res.status, res.statusText, body);
    }
    return res.json();
  },

  async getDemoCase(signal?: AbortSignal): Promise<Record<string, unknown>> {
    return fetchAPI('/v1/qc/demo-case', undefined, signal);
  },

  getTemplateUrl(modality: string): string {
    return `${BASE_URL}/v1/qc/templates/${modality}`;
  },

  getReportPdfUrl(caseId: string): string {
    return `${BASE_URL}/v1/cases/${caseId}/report.pdf`;
  },

  async createBatch(formData?: FormData, useDemo = false, signal?: AbortSignal): Promise<BatchResponse> {
    const url = `${BASE_URL}/v1/batches`;
    let bodyData: BodyInit;
    if (formData) {
      bodyData = formData;
    } else {
      const fd = new FormData();
      if (useDemo) fd.append('use_demo', 'true');
      bodyData = fd;
    }
    const res = await fetch(url, {
      method: 'POST',
      body: bodyData,
      signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
      },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(res.status, res.statusText, body);
    }
    return res.json();
  },

  async getBatch(batchId: string, signal?: AbortSignal): Promise<BatchResponse> {
    return fetchAPI(`/v1/batches/${batchId}`, undefined, signal);
  },

  getBatchResultsUrl(batchId: string): string {
    return `${BASE_URL}/v1/batches/${batchId}/results.csv`;
  },

  async getCase(caseId: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
    return fetchAPI(`/v1/cases/${caseId}`, undefined, signal);
  },

  async getReportPdfBlob(caseId: string, signal?: AbortSignal): Promise<Blob> {
    const url = `${BASE_URL}/v1/cases/${caseId}/report.pdf`;
    const res = await fetch(url, {
      method: 'GET',
      signal,
      headers: {
        'ngrok-skip-browser-warning': 'true',
        'X-API-Key': 'dev-admin-key',
      },
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(res.status, res.statusText, body);
    }
    return res.blob();
  },

  async getSecurityStatus(patientId: string): Promise<PatientSecurityStatus> {
    return fetchAPI<PatientSecurityStatus>(`/api/security/patient/${patientId}`);
  },

  async anchorPatient(patientId: string): Promise<PatientSecurityStatus> {
    return fetchAPI<PatientSecurityStatus>(`/api/security/anchor/${patientId}`, { method: 'POST' });
  },

  async simulateTamper(patientId: string): Promise<Record<string, unknown>> {
    return fetchAPI<Record<string, unknown>>(`/api/security/simulate-tamper/${patientId}`, { method: 'POST' });
  },

  async restorePatient(patientId: string): Promise<Record<string, unknown>> {
    return fetchAPI<Record<string, unknown>>(`/api/security/restore/${patientId}`, { method: 'POST' });
  },

  async getCohortStatus(): Promise<CohortBatchStatus> {
    return fetchAPI<CohortBatchStatus>('/api/security/cohort/status');
  },

  async anchorCohort(): Promise<Record<string, unknown>> {
    return fetchAPI<Record<string, unknown>>('/api/security/cohort/anchor', { method: 'POST' });
  },

  async getPatientMerkleProof(patientId: string): Promise<MerkleProof> {
    return fetchAPI<MerkleProof>(`/api/security/cohort/proof/${patientId}`);
  },

  async verifyPatientMerkle(patientId: string): Promise<Record<string, unknown>> {
    return fetchAPI<Record<string, unknown>>(`/api/security/cohort/verify/${patientId}`, { method: 'POST' });
  },
};

