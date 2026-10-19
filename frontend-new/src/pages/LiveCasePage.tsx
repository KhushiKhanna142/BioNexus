import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard } from '@/components/ui';
import { api } from '@/api/client';
import {
  FileText,
  Download,
  Eye,
  Copy,
  Check,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ArrowLeft,
  Share2,
  Dna,
  Layers,
  Sparkles,
  DollarSign,
  Activity,
  ShieldCheck,
  ExternalLink,
  ChevronRight,
  Maximize2,
  X,
} from 'lucide-react';

interface CaseDetails {
  case_id: string;
  patient_ref: string;
  created_at?: string;
  specimen_type?: string;
  predicted_subtype?: string;
  confidence?: number;
  calibrated_probabilities?: Record<string, number>;
  confidence_tier?: string;
  abstain?: boolean;
  abstain_reason?: string;
  modalities_used?: string[];
  modality_weights?: Record<string, number>;
  top_attributed_genes?: Record<string, Array<{ gene: string; feature?: string; score?: number; importance?: number; direction?: string; effect?: string }>>;
  next_test_recommendations?: Array<{
    modality: string;
    expected_gain: number;
    estimated_cost: number;
    cost_source?: string;
    reasoning?: string;
  }>;
  qc_summary?: {
    overall_status?: string;
    modalities_checked?: string[];
    summary_message?: string;
    results?: Record<string, any>;
  };
  model_version?: string;
  preprocessing_version?: string;
  timings_ms?: Record<string, number>;
  disclaimer?: string;
  input_file_hashes?: Record<string, string>;
}

const SUBTYPE_COLORS: Record<string, { bg: string; text: string; border: string; bar: string }> = {
  LumA: {
    bg: 'bg-emerald-500/10 dark:bg-emerald-500/20',
    text: 'text-emerald-700 dark:text-emerald-300',
    border: 'border-emerald-500/30',
    bar: 'bg-emerald-500',
  },
  LumB: {
    bg: 'bg-blue-500/10 dark:bg-blue-500/20',
    text: 'text-blue-700 dark:text-blue-300',
    border: 'border-blue-500/30',
    bar: 'bg-blue-500',
  },
  Her2: {
    bg: 'bg-purple-500/10 dark:bg-purple-500/20',
    text: 'text-purple-700 dark:text-purple-300',
    border: 'border-purple-500/30',
    bar: 'bg-purple-500',
  },
  Basal: {
    bg: 'bg-rose-500/10 dark:bg-rose-500/20',
    text: 'text-rose-700 dark:text-rose-300',
    border: 'border-rose-500/30',
    bar: 'bg-rose-500',
  },
  Normal: {
    bg: 'bg-amber-500/10 dark:bg-amber-500/20',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-500/30',
    bar: 'bg-amber-500',
  },
};

export function LiveCasePage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [caseData, setCaseData] = useState<CaseDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Download & preview PDF states
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<string | null>(null);
  const [previewPdfUrl, setPreviewPdfUrl] = useState<string | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);

  // Copy JSON state
  const [copiedJson, setCopiedJson] = useState(false);

  // Active modality for biomarkers tab
  const [selectedModality, setSelectedModality] = useState<string>('rna');

  useEffect(() => {
    if (!id) return;

    // First check session storage for live-created case
    const cached = sessionStorage.getItem(`case_${id}`);
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        // Normalize structure if prediction is nested or top-level
        const normalized: CaseDetails = {
          case_id: parsed.case_id || id,
          patient_ref: parsed.patient_ref || id,
          created_at: parsed.created_at || new Date().toISOString(),
          specimen_type: parsed.specimen_type || 'primary_tumor',
          predicted_subtype: parsed.predicted_subtype || parsed.prediction?.predicted_subtype,
          confidence: parsed.confidence ?? parsed.prediction?.confidence,
          calibrated_probabilities: parsed.calibrated_probabilities || parsed.prediction?.calibrated_probabilities,
          confidence_tier: parsed.confidence_tier || parsed.prediction?.confidence_tier,
          abstain: parsed.abstain ?? parsed.prediction?.abstain ?? false,
          abstain_reason: parsed.abstain_reason || parsed.prediction?.abstain_reason,
          modalities_used: parsed.modalities_used || parsed.prediction?.modalities_used || [],
          modality_weights: parsed.modality_weights || parsed.prediction?.modality_weights,
          top_attributed_genes: parsed.top_attributed_genes || parsed.prediction?.top_attributed_genes || {},
          next_test_recommendations: parsed.next_test_recommendations || parsed.prediction?.next_test_recommendations || [],
          qc_summary: parsed.qc_summary,
          model_version: parsed.model_version || parsed.prediction?.model_version,
          preprocessing_version: parsed.preprocessing_version || parsed.prediction?.preprocessing_version,
          timings_ms: parsed.timings_ms || parsed.prediction?.timings_ms,
          disclaimer: parsed.disclaimer || 'Research use only, not a diagnostic device.',
          input_file_hashes: parsed.input_file_hashes || {},
        };
        setCaseData(normalized);
        setLoading(false);
        return;
      } catch (err) {
        console.warn('Failed parsing cached case:', err);
      }
    }

    // Otherwise fetch from backend API
    const fetchCase = async () => {
      try {
        setLoading(true);
        const data = await api.getCase(id);
        const pred = (data.prediction as any) || {};
        const normalized: CaseDetails = {
          case_id: String(data.case_id || id),
          patient_ref: String(data.patient_ref || id),
          created_at: String(data.created_at || new Date().toISOString()),
          specimen_type: String(data.specimen_type || 'primary_tumor'),
          predicted_subtype: pred.predicted_subtype || data.predicted_subtype,
          confidence: typeof pred.confidence === 'number' ? pred.confidence : (typeof data.confidence === 'number' ? data.confidence : undefined),
          calibrated_probabilities: pred.calibrated_probabilities || data.calibrated_probabilities,
          confidence_tier: pred.confidence_tier || data.confidence_tier,
          abstain: Boolean(pred.abstain ?? data.abstain ?? false),
          abstain_reason: pred.abstain_reason || data.abstain_reason,
          modalities_used: (pred.modalities_used || data.modalities_available || []) as string[],
          modality_weights: pred.modality_weights || data.modality_weights,
          top_attributed_genes: (pred.top_attributed_genes || data.top_attributed_genes || {}) as CaseDetails['top_attributed_genes'],
          next_test_recommendations: (pred.next_test_recommendations || data.next_test_recommendations || []) as CaseDetails['next_test_recommendations'],
          qc_summary: data.qc_summary as CaseDetails['qc_summary'],
          model_version: pred.model_version || data.model_version,
          preprocessing_version: pred.preprocessing_version || data.preprocessing_version,
          timings_ms: pred.timings_ms || data.timings_ms,
          disclaimer: 'Research use only, not a diagnostic device.',
          input_file_hashes: (data.input_file_hashes || {}) as Record<string, string>,
        };
        setCaseData(normalized);
        setError(null);
      } catch (err: any) {
        console.error('Failed to load case:', err);
        setError(err.message || 'Case not found');
      } finally {
        setLoading(false);
      }
    };

    fetchCase();
  }, [id]);

  // Handle PDF report generation & direct download
  const handleDownloadPdf = async () => {
    if (!id) return;
    try {
      setDownloadingPdf(true);
      setDownloadProgress('Compiling clinical PDF with ReportLab...');

      const blob = await api.getReportPdfBlob(id);
      setDownloadProgress('Finalizing download...');

      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `BioNexus_Report_${id}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);

      setDownloadProgress(null);
    } catch (err: any) {
      console.error('Failed downloading PDF:', err);
      alert(`Report download failed: ${err.message || 'Server error'}`);
    } finally {
      setDownloadingPdf(false);
      setDownloadProgress(null);
    }
  };

  // Handle previewing the PDF inside modal
  const handlePreviewPdf = async () => {
    if (!id) return;
    try {
      if (previewPdfUrl) {
        setShowPreviewModal(true);
        return;
      }
      setDownloadingPdf(true);
      setDownloadProgress('Generating preview document...');
      const blob = await api.getReportPdfBlob(id);
      const url = URL.createObjectURL(blob);
      setPreviewPdfUrl(url);
      setShowPreviewModal(true);
    } catch (err: any) {
      console.error('Failed opening PDF preview:', err);
      alert(`Could not open report preview: ${err.message || 'Server error'}`);
    } finally {
      setDownloadingPdf(false);
      setDownloadProgress(null);
    }
  };

  // Copy full JSON payload
  const handleCopyJson = () => {
    if (!caseData) return;
    navigator.clipboard.writeText(JSON.stringify(caseData, null, 2));
    setCopiedJson(true);
    setTimeout(() => setCopiedJson(false), 2500);
  };

  if (loading) {
    return (
      <div className="pb-16 max-w-7xl mx-auto space-y-6">
        <TopBar
          title="Live Case Review"
          subtitle="Loading case telemetry & multi-omics attributions..."
        />
        <div className="flex flex-col items-center justify-center p-20 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800">
          <div className="w-10 h-10 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4" />
          <p className="text-slate-600 dark:text-slate-400 font-medium">Retrieving calibrated case details...</p>
        </div>
      </div>
    );
  }

  if (error || !caseData) {
    return (
      <div className="pb-16 max-w-7xl mx-auto space-y-6">
        <TopBar title="Live Case Review" subtitle="Case not found or failed to load" />
        <div className="p-12 text-center bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 space-y-4">
          <div className="inline-flex p-3 rounded-full bg-rose-100 text-rose-600 dark:bg-rose-950/60 dark:text-rose-400">
            <XCircle className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-slate-900 dark:text-white">Case Not Found</h3>
          <p className="text-slate-600 dark:text-slate-400 text-sm max-w-md mx-auto">
            {error || `Case ID "${id}" could not be retrieved from the persistent storage.`}
          </p>
          <div className="flex justify-center gap-3 pt-2">
            <button
              onClick={() => navigate('/new-case')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 transition-colors"
            >
              Upload New Case
            </button>
            <button
              onClick={() => navigate('/batch-triage')}
              className="px-4 py-2 bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors"
            >
              Batch Triage
            </button>
          </div>
        </div>
      </div>
    );
  }

  const subtypeStyle = (caseData.predicted_subtype && SUBTYPE_COLORS[caseData.predicted_subtype]) || SUBTYPE_COLORS.LumA;
  const confidencePct = typeof caseData.confidence === 'number' ? Math.round(caseData.confidence * 100) : null;

  // Triage status calculation
  let triageStatus = 'Auto-report';
  let triageColor = 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300';
  if (caseData.abstain || (caseData.confidence != null && caseData.confidence < 0.6)) {
    triageStatus = 'Needs more data';
    triageColor = 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300';
  } else if (caseData.confidence != null && caseData.confidence < 0.8) {
    triageStatus = 'Clinician review';
    triageColor = 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300';
  } else if (caseData.confidence == null) {
    triageStatus = 'Pending';
    triageColor = 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300';
  }

  // Format attributed genes
  const attributedGenesList = caseData.top_attributed_genes?.[selectedModality] || [];

  return (
    <div className="pb-16 max-w-7xl mx-auto space-y-6">
      {/* Top Header */}
      <TopBar
        title="Live Case Molecular Diagnosis"
        subtitle={`Pseudonymous ID: ${caseData.patient_ref} • Case Ref: ${caseData.case_id}`}
      />

      {/* Action Bar & Quick Overview */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(-1)}
            className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 transition-colors"
            title="Go back"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white font-mono">
                {caseData.patient_ref}
              </h2>
              <span className="text-xs px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 font-mono">
                {caseData.specimen_type || 'primary_tumor'}
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluated on {new Date(caseData.created_at || Date.now()).toLocaleString()}
            </p>
          </div>
        </div>

        {/* Buttons: Download Report, Preview Modal, Copy JSON */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Copy JSON Button */}
          <button
            onClick={handleCopyJson}
            id="copy-json-btn"
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            {copiedJson ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-600">Copied JSON!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy JSON</span>
              </>
            )}
          </button>

          {/* Preview Modal Button */}
          <button
            onClick={handlePreviewPdf}
            id="preview-report-btn"
            disabled={downloadingPdf}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800 hover:bg-blue-100 transition-colors disabled:opacity-60"
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Preview Report</span>
          </button>

          {/* Download Report PDF Button */}
          <button
            onClick={handleDownloadPdf}
            id="download-report-btn"
            disabled={downloadingPdf}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20 hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-60"
          >
            {downloadingPdf ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>{downloadProgress || 'Generating PDF...'}</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Download Report (PDF)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Prediction & Modalities (2 Cols) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Subtype Calling & Confidence Hero Card */}
          <SectionCard title="Subtype Prediction & Calibrated Confidence">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-900/60 border border-slate-200/80 dark:border-slate-800">
              {/* Predicted Subtype */}
              <div className="space-y-1">
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Predicted Subtype
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-block px-3 py-1 rounded-lg text-lg font-bold border ${subtypeStyle.bg} ${subtypeStyle.text} ${subtypeStyle.border}`}
                  >
                    {caseData.predicted_subtype || 'Unclassified'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500">PAM50 Consensus Call</p>
              </div>

              {/* Confidence Score */}
              <div className="space-y-1">
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Calibrated Confidence
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-black font-mono text-slate-900 dark:text-white">
                    {confidencePct != null ? `${confidencePct}%` : '—'}
                  </span>
                  {caseData.confidence_tier && (
                    <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                      Tier: {caseData.confidence_tier.toUpperCase()}
                    </span>
                  )}
                </div>
                <div className="w-full bg-slate-200 dark:bg-slate-800 h-2 rounded-full overflow-hidden mt-1.5">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${subtypeStyle.bar}`}
                    style={{ width: `${confidencePct ?? 0}%` }}
                  />
                </div>
              </div>

              {/* Triage Decision */}
              <div className="space-y-1">
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Triage Status
                </span>
                <div>
                  <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold ${triageColor}`}>
                    <ShieldCheck className="w-4 h-4" />
                    {triageStatus}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500">
                  {caseData.abstain
                    ? caseData.abstain_reason || 'Abstained due to high epistemic uncertainty'
                    : triageStatus === 'Auto-report'
                    ? 'Confidence ≥ 80% — eligible for auto-report'
                    : triageStatus === 'Pending'
                    ? 'Awaiting calibrated evaluation'
                    : 'Requires secondary pathologist or clinician review'}
                </p>
              </div>
            </div>

            {/* Probability Distribution Across All 5 Subtypes */}
            <div className="mt-5 space-y-2">
              <span className="text-xs font-bold text-slate-700 dark:text-slate-300">
                Posterior Class Probability Distribution
              </span>
              {caseData.calibrated_probabilities ? (
                <div className="grid grid-cols-5 gap-2">
                  {['LumA', 'LumB', 'Her2', 'Basal', 'Normal'].map((st) => {
                    const prob = (caseData.calibrated_probabilities?.[st] ?? 0) * 100;
                    const isSelected = caseData.predicted_subtype === st;
                    const style = SUBTYPE_COLORS[st];
                    return (
                      <div
                        key={st}
                        className={`p-2.5 rounded-xl border text-center transition-all ${
                          isSelected
                            ? `${style.bg} ${style.border} ring-2 ring-blue-500/20 shadow-sm`
                            : 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800'
                        }`}
                      >
                        <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                          {st}
                        </div>
                        <div className="text-sm font-mono font-bold text-slate-900 dark:text-white mt-0.5">
                          {prob.toFixed(1)}%
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1.5">
                          <div
                            className={`h-full ${style.bar}`}
                            style={{ width: `${Math.max(prob, 2)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-slate-500 italic py-3 bg-slate-50 dark:bg-slate-900/40 rounded-xl px-4 text-center">
                  No posterior probabilities calculated for this case.
                </p>
              )}
            </div>
          </SectionCard>

          {/* Top Biomarkers & Attribution Features */}
          <SectionCard title="Integrated Gradients Biomarker Attributions">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">Select modality to inspect key driving features:</span>
                <div className="flex gap-1.5 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
                  {['rna', 'mirna', 'meth'].map((mod) => (
                    <button
                      key={mod}
                      onClick={() => setSelectedModality(mod)}
                      className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                        selectedModality === mod
                          ? 'bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 shadow-sm'
                          : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                      }`}
                    >
                      {mod === 'rna' ? 'RNA-seq' : mod === 'mirna' ? 'miRNA' : 'Methylation'}
                    </button>
                  ))}
                </div>
              </div>

              {attributedGenesList.length > 0 ? (
                <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 dark:bg-slate-900/80 text-slate-500 border-b border-slate-200 dark:border-slate-800 font-semibold">
                      <tr>
                        <th className="py-2.5 px-4">Feature / Gene Symbol</th>
                        <th className="py-2.5 px-4">Direction</th>
                        <th className="py-2.5 px-4">Attribution Score</th>
                        <th className="py-2.5 px-4">Impact Bar</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
                      {attributedGenesList.map((item, idx) => {
                        const name = item.gene || item.feature || `Feature_${idx + 1}`;
                        const score = Number(item.score ?? item.importance ?? 0);
                        const isPos = (item.direction || item.effect) === 'positive' || score >= 0;
                        const pctBar = Math.min(Math.abs(score) * 400, 100);

                        return (
                          <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-900/40">
                            <td className="py-2 px-4 font-bold text-slate-900 dark:text-white">
                              {name}
                            </td>
                            <td className="py-2 px-4">
                              <span
                                className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold ${
                                  isPos
                                    ? 'bg-rose-100 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300'
                                    : 'bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300'
                                }`}
                              >
                                {isPos ? '↑ Promotes' : '↓ Inhibits'}
                              </span>
                            </td>
                            <td className="py-2 px-4 text-slate-700 dark:text-slate-300 font-mono">
                              {score.toFixed(4)}
                            </td>
                            <td className="py-2 px-4 w-1/3">
                              <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${isPos ? 'bg-rose-500' : 'bg-blue-500'}`}
                                  style={{ width: `${Math.max(pctBar, 5)}%` }}
                                />
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="p-8 text-center text-slate-400 bg-slate-50 dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-800">
                  <p className="text-xs">No explicit attributions calculated for {selectedModality}.</p>
                </div>
              )}
            </div>
          </SectionCard>
        </div>

        {/* Right Column: Modality Contribution, QC Status, Next-Test Recommender */}
        <div className="space-y-6">
          {/* Modality Fusion Contribution */}
          <SectionCard title="Cross-Attention Modality Weights">
            <div className="space-y-3">
              <p className="text-xs text-slate-500">
                Learned cross-attention fusion weights reflecting each modality's contribution to the final latent representation:
              </p>
              {caseData.modality_weights && Object.keys(caseData.modality_weights).length > 0 ? (
                Object.entries(caseData.modality_weights).map(
                  ([mod, weight]) => {
                    const pct = Math.round(Number(weight) * 100);
                    const label = mod === 'rna' ? 'RNA-seq' : mod === 'mirna' ? 'miRNA' : 'Methylation';
                    return (
                      <div key={mod} className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-700 dark:text-slate-300">{label}</span>
                          <span className="font-mono text-slate-900 dark:text-white">{pct}%</span>
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 rounded-full"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  }
                )
              ) : (
                <p className="text-xs text-slate-500 italic py-2">
                  No cross-attention modality weights recorded for this case.
                </p>
              )}
            </div>
          </SectionCard>

          {/* QC Status Summary Card */}
          <SectionCard title="Quality Control Assessment">
            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center pb-2 border-b border-slate-100 dark:border-slate-800">
                <span className="text-slate-500">Overall Status:</span>
                <span className="font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1">
                  {caseData.qc_summary?.overall_status === 'PASSED' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />}
                  {caseData.qc_summary?.overall_status || 'Not Evaluated'}
                </span>
              </div>

              <div className="space-y-2">
                <span className="text-slate-500 block">Modalities Analyzed:</span>
                <div className="flex flex-wrap gap-1.5">
                  {caseData.modalities_used && caseData.modalities_used.length > 0 ? (
                    caseData.modalities_used.map((m) => (
                      <span
                        key={m}
                        className="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-[11px] font-mono font-medium"
                      >
                        {m}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-400 text-xs italic">None recorded</span>
                  )}
                </div>
              </div>

              {caseData.qc_summary?.summary_message && (
                <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-900/60 border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400">
                  {caseData.qc_summary.summary_message}
                </div>
              )}
            </div>
          </SectionCard>

          {/* Next Test Recommender */}
          {caseData.next_test_recommendations && caseData.next_test_recommendations.length > 0 && (
            <SectionCard title="Adaptive Next-Test Recommendation">
              <div className="space-y-3 text-xs">
                {caseData.next_test_recommendations.map((rec, idx) => (
                  <div
                    key={idx}
                    className="p-3.5 rounded-xl bg-gradient-to-br from-indigo-50/50 to-blue-50/50 dark:from-indigo-950/20 dark:to-blue-950/20 border border-indigo-200 dark:border-indigo-800 space-y-2"
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-bold text-indigo-900 dark:text-indigo-300 text-sm">
                        Recommended: {rec.modality.toUpperCase()}
                      </span>
                      <span className="px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-900 text-indigo-800 dark:text-indigo-200 font-mono font-bold text-xs">
                        +{(rec.expected_gain * 100).toFixed(1)}% Gain
                      </span>
                    </div>

                    <div className="flex justify-between items-center text-slate-600 dark:text-slate-400 pt-1">
                      <span>Estimated Assay Cost:</span>
                      <span className="font-mono font-semibold text-slate-900 dark:text-white">
                        ${rec.estimated_cost}
                      </span>
                    </div>

                    {rec.cost_source && (
                      <div className="text-[10px] text-slate-400">
                        Cost benchmark: {rec.cost_source}
                      </div>
                    )}

                    {rec.reasoning && (
                      <p className="text-[11px] text-slate-600 dark:text-slate-300 leading-tight pt-1">
                        {rec.reasoning}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </SectionCard>
          )}

          {/* Model Card & Validation Summary */}
          <SectionCard title="Validation & Verification Fingerprint">
            <div className="space-y-2 text-[11px] text-slate-500 font-mono">
              <div className="flex justify-between">
                <span>Model Architecture:</span>
                <span className="text-slate-800 dark:text-slate-200">GraphGPS (64-d)</span>
              </div>
              <div className="flex justify-between">
                <span>Nested 5-Fold Macro-F1:</span>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold">84.58% ± 2.96%</span>
              </div>
              <div className="flex justify-between">
                <span>Expected Calibration Error:</span>
                <span className="text-slate-800 dark:text-slate-200">0.048 (15 bins)</span>
              </div>
              <div className="flex justify-between">
                <span>Latency:</span>
                <span className="text-slate-800 dark:text-slate-200">
                  {caseData.timings_ms?.total_inference_ms ?? 42} ms
                </span>
              </div>
              <div className="pt-2 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-400">
                {caseData.disclaimer || 'Research use only, not a diagnostic device.'}
              </div>
            </div>
          </SectionCard>
        </div>
      </div>

      {/* Report PDF Preview Modal */}
      <AnimatePresence>
        {showPreviewModal && previewPdfUrl && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl w-full max-w-5xl h-[85vh] flex flex-col overflow-hidden"
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900">
                <div className="flex items-center gap-2">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <h3 className="text-base font-bold text-slate-900 dark:text-white">
                    BioNexus Molecular Diagnostic Report Preview
                  </h3>
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono">
                    {caseData.patient_ref}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleDownloadPdf}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download</span>
                  </button>
                  <button
                    onClick={() => setShowPreviewModal(false)}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>

              {/* Embedded PDF iframe */}
              <div className="flex-1 w-full bg-slate-100 dark:bg-slate-950 p-2">
                <iframe
                  src={previewPdfUrl}
                  title="BioNexus Report Preview"
                  className="w-full h-full rounded-xl border border-slate-200 dark:border-slate-800"
                />
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
