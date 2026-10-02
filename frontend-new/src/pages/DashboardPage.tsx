import { useState, useEffect, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Users, Layers, GitBranch, TrendingUp, ArrowRight, Play,
  CheckCircle2, AlertCircle, Info, Sparkles, Lightbulb,
  ArrowUpRight, ArrowDownRight, RefreshCw, UploadCloud,
} from 'lucide-react';
import { TopBar } from '@/components/layout/TopBar';
import { api } from '@/api/client';
import { useHealth } from '@/lib/health';
import type { Patient, PAM50Subtype, PredictionResult, ExplanationResult, PatientEgoGraphResponse } from '@/lib/types';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

// Subtype Styling Definitions
const SUBTYPE_THEME: Record<string, { bg: string; text: string; hex: string; lightHex: string; desc: string }> = {
  LumA: {
    bg: 'bg-blue-50',
    text: 'text-blue-600',
    hex: '#3B82F6',
    lightHex: '#93C5FD',
    desc: 'Luminal A subtype is typically associated with hormone receptor positivity and better prognosis.',
  },
  LumB: {
    bg: 'bg-pink-50',
    text: 'text-pink-600',
    hex: '#EC4899',
    lightHex: '#F472B6',
    desc: 'Luminal B tumors tend to be ER-positive but often have higher histological grade and proliferative index.',
  },
  Her2: {
    bg: 'bg-amber-50',
    text: 'text-amber-600',
    hex: '#F59E0B',
    lightHex: '#FCD34D',
    desc: 'HER2-enriched subtype exhibits high expression of the ERBB2 amplicon and targeted therapy responsiveness.',
  },
  Basal: {
    bg: 'bg-purple-50',
    text: 'text-purple-600',
    hex: '#8B5CF6',
    lightHex: '#C4B5FD',
    desc: 'Basal-like subtype is frequently triple-negative (ER-/PR-/HER2-) with high genomic instability.',
  },
  Normal: {
    bg: 'bg-emerald-50',
    text: 'text-emerald-600',
    hex: '#10B981',
    lightHex: '#6EE7B7',
    desc: 'Normal-like breast tumors share gene expression characteristics with non-malignant breast tissue.',
  },
};

const SUBTYPE_COLORS: Record<string, string> = {
  LumA: '#3B82F6',
  LumB: '#EC4899',
  Her2: '#F59E0B',
  Basal: '#8B5CF6',
  Normal: '#10B981',
};

export function DashboardPage() {
  const health = useHealth();
  const navigate = useNavigate();

  // Patients State
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>('TCGA-AR-A1AR');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  // Cohort & Metrics State
  const [cohortStats, setCohortStats] = useState<Awaited<ReturnType<typeof api.getCohort>> | null>(null);
  const [modelMetrics, setModelMetrics] = useState<Awaited<ReturnType<typeof api.getMetrics>> | null>(null);

  // Clinical Details State (real from clinical file)
  const [patientClinical, setPatientClinical] = useState<Record<string, unknown> | null>(null);

  // Prediction State
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Explanation / Biomarkers State
  const [biomarkers, setBiomarkers] = useState<ExplanationResult['biomarkers']>([]);
  const [isLoadingBiomarkers, setIsLoadingBiomarkers] = useState<boolean>(false);

  // Real Multi-Omics Heatmap State
  const [activeProfileLayer, setActiveProfileLayer] = useState<'rna' | 'mirna' | 'meth'>('rna');
  const [heatmapData, setHeatmapData] = useState<{
    layer: string;
    patient_ids: string[];
    feature_ids: string[];
    symbols: string[];
    z_scores: number[][];
  } | null>(null);
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState<boolean>(false);
  const [hoveredCell, setHoveredCell] = useState<{ name: string; sampleIdx: number; value: number } | null>(null);

  // Cohort Analysis Tab & Real Embeddings State
  const [cohortTab, setCohortTab] = useState<'distribution' | 'pca' | 'tsne' | 'survival'>('distribution');
  const [embeddingsPoints, setEmbeddingsPoints] = useState<{ patient_id: string; true_pam50: string; x: number; y: number }[]>([]);

  // Model Insights Tab & Real Ego Graph State
  const [insightsTab, setInsightsTab] = useState<'graph' | 'attention' | 'pathway'>('graph');
  const [egoGraph, setEgoGraph] = useState<PatientEgoGraphResponse | null>(null);

  // Initial Data Fetch
  useEffect(() => {
    api.getCohort().then(setCohortStats).catch(console.error);
    api.getMetrics().then(setModelMetrics).catch(console.error);

    api.getPatients()
      .then((pts) => {
        setPatients(pts);
        if (pts.length > 0) {
          const found = pts.find((p) => p.id === 'TCGA-AR-A1AR');
          if (found) {
            setSelectedPatientId(found.id);
            setSelectedPatient(found);
          } else {
            setSelectedPatientId(pts[0].id);
            setSelectedPatient(pts[0]);
          }
        }
      })
      .catch(console.error);

    // Load UMAP embeddings for cohort visualization
    api.getEmbeddings('fused', 'umap')
      .then((res) => {
        setEmbeddingsPoints(res.points || []);
      })
      .catch(console.error);
  }, []);

  // When selectedPatientId changes, run prediction & fetch clinical & explanation
  useEffect(() => {
    if (!selectedPatientId) return;

    const patientObj = patients.find((p) => p.id === selectedPatientId);
    if (patientObj) setSelectedPatient(patientObj);

    // Fetch real clinical data
    api.getPatientClinical(selectedPatientId)
      .then(setPatientClinical)
      .catch(() => setPatientClinical(null));

    setIsPredicting(true);
    setPredictError(null);

    // Run real prediction
    api.predict(selectedPatientId)
      .then((res) => {
        setPrediction(res);
      })
      .catch((err) => {
        console.error('Prediction error:', err);
        setPredictError(err.message || 'Failed to fetch out-of-fold prediction.');
        setPrediction(null);
      })
      .finally(() => setIsPredicting(false));

    // Fetch biomarkers
    setIsLoadingBiomarkers(true);
    api.getExplanation(selectedPatientId, 5, 'rna')
      .then((exp) => {
        setBiomarkers(exp.biomarkers || []);
      })
      .catch(() => {
        setBiomarkers([]);
      })
      .finally(() => setIsLoadingBiomarkers(false));

    // Fetch real ego graph for model insights
    api.getPatientEgoGraph('rna', selectedPatientId, 1)
      .then(setEgoGraph)
      .catch(() => setEgoGraph(null));
  }, [selectedPatientId, patients]);

  // Fetch real heatmap when activeProfileLayer changes
  useEffect(() => {
    setIsLoadingHeatmap(true);
    api.getHeatmap(activeProfileLayer, undefined, undefined, 8)
      .then((res) => {
        setHeatmapData(res);
      })
      .catch((err) => {
        console.error('Heatmap load error:', err);
        setHeatmapData(null);
      })
      .finally(() => setIsLoadingHeatmap(false));
  }, [activeProfileLayer]);

  const activeSubtype = prediction?.predicted_subtype || selectedPatient?.true_pam50 || 'LumA';
  const theme = SUBTYPE_THEME[activeSubtype] || SUBTYPE_THEME.LumA;

  // Subtype Distribution Data for Cohort Chart (real counts only, no fallbacks)
  const subtypeDistData = useMemo(() => {
    if (cohortStats?.subtype_distribution) {
      return [
        { name: 'LumA', count: cohortStats.subtype_distribution.LumA || 0, fill: '#3B82F6' },
        { name: 'LumB', count: cohortStats.subtype_distribution.LumB || 0, fill: '#EC4899' },
        { name: 'Her2', count: cohortStats.subtype_distribution.Her2 || 0, fill: '#F59E0B' },
        { name: 'Basal', count: cohortStats.subtype_distribution.Basal || 0, fill: '#8B5CF6' },
        { name: 'Normal', count: cohortStats.subtype_distribution.Normal || 0, fill: '#10B981' },
      ];
    }
    return [];
  }, [cohortStats]);

  // Normalized embedding bounds for scatter SVG
  const scatterBounds = useMemo(() => {
    if (!embeddingsPoints.length) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    embeddingsPoints.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    return { minX, maxX, minY, maxY, spanX, spanY };
  }, [embeddingsPoints]);

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      <TopBar
        title="Welcome to BioNexus"
        subtitle="Multi-Omics Graph Transformer for Breast Cancer Subtype Classification"
        onSelectPatient={(id) => setSelectedPatientId(id)}
      />

      <main className="p-6 max-w-[1720px] mx-auto space-y-6">
        {/* ========================================================================= */}
        {/* TOP SECTION: Left 65% (Stats + Hero) | Right 35% (Patient Analysis Panel) */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column (approx 60-65%) */}
          <div className="lg:col-span-7 xl:col-span-8 space-y-5 flex flex-col justify-between">
            {/* 4 Stat Cards Row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {/* Card 1: Patients in Cohort */}
              <div className="bg-white rounded-2xl p-4 border border-slate-100 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center shrink-0">
                    <Users className="w-5 h-5" />
                  </div>
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {cohortStats?.total_patients != null ? cohortStats.total_patients : '—'}
                  </span>
                </div>
                <div className="text-xs font-medium text-slate-500">Patients in Cohort</div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">TCGA-BRCA Multi-Omics</div>
              </div>

              {/* Card 2: Omics Layers */}
              <div className="bg-white rounded-2xl p-4 border border-slate-100 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-9 h-9 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center shrink-0">
                    <Layers className="w-5 h-5" />
                  </div>
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">3</span>
                </div>
                <div className="text-xs font-medium text-slate-500">Omics Layers</div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">RNA • miRNA • Methylation</div>
              </div>

              {/* Card 3: PAM50 Subtypes */}
              <div className="bg-white rounded-2xl p-4 border border-slate-100 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-9 h-9 rounded-xl bg-teal-50 text-teal-600 flex items-center justify-center shrink-0">
                    <GitBranch className="w-5 h-5" />
                  </div>
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">5</span>
                </div>
                <div className="text-xs font-medium text-slate-500">PAM50 Subtypes</div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">LumA • LumB • Her2 • Basal • Normal</div>
              </div>

              {/* Card 4: Best Model Accuracy */}
              <div className="bg-white rounded-2xl p-4 border border-slate-100 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:shadow-md transition-shadow">
                <div className="flex items-center gap-3 mb-2">
                  <div className="w-9 h-9 rounded-xl bg-sky-50 text-sky-600 flex items-center justify-center shrink-0">
                    <TrendingUp className="w-5 h-5" />
                  </div>
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {modelMetrics?.accuracy != null ? `${(modelMetrics.accuracy * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
                <div className="text-xs font-medium text-slate-500">Best Model Accuracy</div>
                <div className="text-[10px] text-slate-400 truncate mt-0.5">5-fold cross-validation within TCGA-BRCA</div>
              </div>
            </div>

            {/* Hero Banner with Integrated Biological Molecular Background */}
            <div className="relative rounded-2xl bg-gradient-to-r from-[#EFF4FE] via-[#F3EAFD] to-[#E9EFFD] border border-blue-100/70 p-7 overflow-hidden flex-1 flex flex-col justify-center min-h-[220px] shadow-[0_2px_8px_rgba(59,99,224,0.04)]">
              {/* Molecular & DNA Illustration in Right Background */}
              <div
                className="absolute right-0 top-0 bottom-0 w-1/2 bg-contain bg-no-repeat bg-right pointer-events-none opacity-95 mix-blend-multiply"
                style={{
                  backgroundImage: "url('/hero-molecules.png')",
                  maskImage: 'linear-gradient(to left, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%)',
                  WebkitMaskImage: 'linear-gradient(to left, rgba(0,0,0,1) 50%, rgba(0,0,0,0) 100%)',
                }}
              />

              {/* Content */}
              <div className="relative z-10 max-w-md sm:max-w-lg space-y-3">
                <h2 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight leading-snug">
                  Integrating Multiple Molecular Layers for Smarter Cancer Subtyping
                </h2>
                <p className="text-xs sm:text-sm text-slate-500 font-medium tracking-wide">
                  Explore • Predict • Interpret • Advance Research
                </p>
                <div className="flex flex-wrap items-center gap-3 pt-2">
                  <Link
                    to="/patient-analysis"
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#0B1B3F] hover:bg-[#1A2D5A] text-white text-xs font-semibold shadow-sm transition-all duration-150 hover:gap-2.5"
                  >
                    <span>Analyze a Patient</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                  <Link
                    to="/data-explorer"
                    className="inline-flex items-center px-5 py-2.5 rounded-full bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold border border-slate-200/90 shadow-2xs transition-colors"
                  >
                    Explore Dataset
                  </Link>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Patient Analysis Card (approx 35-40%) */}
          <div className="lg:col-span-5 xl:col-span-4 bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            {/* Header + New Patient Button */}
            <div>
              <div className="flex items-start justify-between gap-2 mb-1">
                <div>
                  <h3 className="font-bold text-slate-900 text-base">Patient Analysis</h3>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Select a cohort patient to view out-of-fold PAM50 prediction and multi-omics profile.
                  </p>
                </div>
                <button
                  onClick={() => navigate('/patient-analysis')}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0B1B3F] hover:bg-[#1A2D5A] text-white text-[11px] font-semibold transition-colors shrink-0 shadow-2xs"
                >
                  <span>+ New Patient</span>
                </button>
              </div>

              {/* Patient Selection & Run Prediction Bar */}
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-2 mt-3.5">
                <div className="sm:col-span-7">
                  <select
                    value={selectedPatientId}
                    onChange={(e) => setSelectedPatientId(e.target.value)}
                    className="w-full h-9 px-3 bg-slate-50 border border-slate-200 rounded-xl text-xs font-mono font-medium text-slate-800 outline-none focus:border-blue-500 focus:bg-white transition-all cursor-pointer"
                  >
                    {patients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.id}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="sm:col-span-5">
                  <button
                    onClick={() => {
                      setIsPredicting(true);
                      setPredictError(null);
                      api.predict(selectedPatientId)
                        .then(setPrediction)
                        .catch((err) => setPredictError(err.message || 'Prediction failed'))
                        .finally(() => setIsPredicting(false));
                    }}
                    disabled={isPredicting}
                    className="w-full h-9 inline-flex items-center justify-center gap-1.5 px-3 rounded-xl bg-[#0B1B3F] hover:bg-[#1A2D5A] disabled:opacity-75 text-white text-xs font-medium transition-colors shadow-2xs"
                  >
                    {isPredicting ? (
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Play className="w-3 h-3 fill-current" />
                    )}
                    <span>Run Prediction</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              </div>

              {/* Patient Clinical Info Pills (Real from clinical file) */}
              <div className="grid grid-cols-5 gap-1.5 mt-3.5 pt-3 border-t border-slate-100 text-center">
                <div className="bg-slate-50/80 rounded-lg p-1.5">
                  <div className="text-[9.5px] uppercase tracking-wider text-slate-400 font-semibold">Patient ID</div>
                  <div className="text-[11px] font-mono font-bold text-slate-800 truncate" title={selectedPatientId}>
                    {selectedPatientId.replace('TCGA-', '')}
                  </div>
                </div>
                <div className="bg-slate-50/80 rounded-lg p-1.5">
                  <div className="text-[9.5px] uppercase tracking-wider text-slate-400 font-semibold">Age</div>
                  <div className="text-[11px] font-bold text-slate-800">
                    {patientClinical?.age != null ? String(patientClinical.age) : 'Not available'}
                  </div>
                </div>
                <div className="bg-slate-50/80 rounded-lg p-1.5">
                  <div className="text-[9.5px] uppercase tracking-wider text-slate-400 font-semibold">ER Status</div>
                  <div className="text-[11px] font-bold text-blue-600 truncate">
                    {patientClinical?.er_status != null ? String(patientClinical.er_status) : 'Not available'}
                  </div>
                </div>
                <div className="bg-slate-50/80 rounded-lg p-1.5">
                  <div className="text-[9.5px] uppercase tracking-wider text-slate-400 font-semibold">Grade</div>
                  <div className="text-[11px] font-bold text-slate-800 truncate">
                    {patientClinical?.neoplasm_histologic_grade != null ? String(patientClinical.neoplasm_histologic_grade) : 'Not available'}
                  </div>
                </div>
                <div className="bg-slate-50/80 rounded-lg p-1.5">
                  <div className="text-[9.5px] uppercase tracking-wider text-slate-400 font-semibold">Lymph Node</div>
                  <div className="text-[11px] font-bold text-slate-800 truncate">
                    {patientClinical?.lymph_nodes_examined_positive != null ? String(patientClinical.lymph_nodes_examined_positive) : 'Not available'}
                  </div>
                </div>
              </div>
            </div>

            {/* Stepper + Anatomy Inset */}
            <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-12 gap-3 items-center">
              {/* Stepper (7 cols) */}
              <div className="col-span-7 space-y-2">
                {[
                  { step: 1, title: 'Patient Information', sub: 'Clinical fields from cohort', active: true },
                  { step: 2, title: 'Multi-Omics Profiles', sub: 'Standardized z-scores', active: false },
                  { step: 3, title: 'Prediction Result', sub: 'OOF subtype and confidence', active: false },
                  { step: 4, title: 'Interpretation', sub: 'Integrated Gradients & ablation', active: false },
                ].map((s) => (
                  <div key={s.step} className="flex items-center gap-2.5">
                    <div
                      className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold shrink-0 ${
                        s.step === 1
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-100 text-slate-400'
                      }`}
                    >
                      {s.step}
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold text-slate-800 leading-tight">{s.title}</div>
                      <div className="text-[9.5px] text-slate-400">{s.sub}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Anatomy Inset Illustration (5 cols) */}
              <div className="col-span-5 flex flex-col items-center">
                <div className="w-full h-24 rounded-xl overflow-hidden bg-gradient-to-br from-pink-50/50 to-blue-50/30 flex items-center justify-center border border-pink-100/50 p-1">
                  <img
                    src="/breast-anatomy.png"
                    alt="Breast Anatomy Cross-section"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            </div>

            {/* About Box Callout */}
            <div className="mt-3 p-2.5 rounded-xl bg-slate-50/90 border border-slate-100 flex items-start gap-2 text-[10.5px] text-slate-500 leading-relaxed">
              <span className="font-semibold text-slate-700 shrink-0">Model:</span>
              <span>
                Predicts PAM50 subtype from patient-similarity kNN graphs (k=10) with GraphGPS encoders and cross-attention fusion.
              </span>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* MIDDLE SECTION (3 Columns): Prediction Result | Multi-Omics Profile | Key Biomarkers */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 1. Prediction Result Card */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-bold text-slate-900 text-base">Prediction Result</h3>
                {prediction ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200/60 text-[11px] font-semibold text-emerald-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    Prediction Complete
                  </span>
                ) : isPredicting ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-blue-50 border border-blue-200/60 text-[11px] font-semibold text-blue-700">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    Running OOF Model...
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 text-[11px] font-medium text-slate-500">
                    Ready
                  </span>
                )}
              </div>

              {predictError && (
                <div className="p-3 mb-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
                  {predictError}
                </div>
              )}

              {prediction ? (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Left: Main Predicted Subtype Display */}
                  <div className="space-y-2">
                    <div className="text-[11px] font-medium text-slate-400">Predicted Subtype</div>
                    <div className="flex items-baseline gap-2">
                      <span className={`text-3xl font-extrabold ${theme.text}`}>
                        {prediction.predicted_subtype}
                      </span>
                      <span className="text-xs text-slate-400 font-medium">Confidence</span>
                      <span className="text-sm font-bold text-slate-800">
                        {(prediction.confidence * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden mt-1">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${Math.round(prediction.confidence * 100)}%`,
                          backgroundColor: theme.hex,
                        }}
                      />
                    </div>
                    <p className="text-[11px] text-slate-500 leading-relaxed mt-2.5">
                      {theme.desc}
                    </p>
                    {prediction.true_pam50 && (
                      <div className="mt-2 text-xs font-medium text-slate-700">
                        True Subtype: <span className="font-bold">{prediction.true_pam50}</span>
                        <span className={`ml-2 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                          prediction.match ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                        }`}>
                          {prediction.match ? 'Match' : 'Mismatch'}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Right: Subtype Probabilities Breakdown */}
                  <div className="space-y-2">
                    <div className="text-[11px] font-medium text-slate-400">Subtype Probabilities</div>
                    <div className="space-y-1.5">
                      {(['LumA', 'LumB', 'Her2', 'Basal', 'Normal'] as PAM50Subtype[]).map((name) => {
                        const prob = prediction.probabilities?.[name] ?? 0;
                        return (
                          <div key={name} className="flex items-center gap-2 text-xs">
                            <span className="w-12 font-medium text-slate-700 text-[11px]">{name}</span>
                            <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all duration-300"
                                style={{
                                  width: `${Math.max(prob * 100, 2)}%`,
                                  backgroundColor: SUBTYPE_COLORS[name],
                                }}
                              />
                            </div>
                            <span className="w-8 text-right font-mono text-[11px] text-slate-600">
                              {(prob * 100).toFixed(0)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center text-slate-400 text-xs">
                  Select a patient and click "Run Prediction" to load real out-of-fold inference.
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>
                {prediction?.fold_id != null ? `Fold ${prediction.fold_id} • Out-of-fold evaluated` : 'Out-of-fold evaluated'}
              </span>
              <Link to="/patient-analysis" className="text-blue-600 font-semibold hover:underline flex items-center gap-1">
                Full details →
              </Link>
            </div>
          </div>

          {/* 2. Multi-Omics Profile Card (Real Heatmap Z-scores) */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-slate-900 text-base">Multi-Omics Profile</h3>
              </div>

              {/* Layer Navigation Pills */}
              <div className="flex items-center gap-1.5 p-1 bg-slate-50 rounded-xl mb-3 border border-slate-100">
                <button
                  onClick={() => setActiveProfileLayer('rna')}
                  className={`flex-1 py-1 px-2 rounded-lg text-[11px] font-semibold transition-all ${
                    activeProfileLayer === 'rna'
                      ? 'bg-[#0B1B3F] text-white shadow-2xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Gene Expression (RNA)
                </button>
                <button
                  onClick={() => setActiveProfileLayer('mirna')}
                  className={`flex-1 py-1 px-2 rounded-lg text-[11px] font-semibold transition-all ${
                    activeProfileLayer === 'mirna'
                      ? 'bg-[#0B1B3F] text-white shadow-2xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  miRNA Expression
                </button>
                <button
                  onClick={() => setActiveProfileLayer('meth')}
                  className={`flex-1 py-1 px-2 rounded-lg text-[11px] font-semibold transition-all ${
                    activeProfileLayer === 'meth'
                      ? 'bg-[#0B1B3F] text-white shadow-2xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  DNA Methylation
                </button>
              </div>

              <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700 mb-2">
                <span>Top Variable {activeProfileLayer === 'rna' ? 'Genes' : activeProfileLayer === 'mirna' ? 'miRNAs' : 'CpG Sites'}</span>
                {hoveredCell && (
                  <span className="text-[10px] font-mono text-blue-600 font-normal">
                    {hoveredCell.name} (S{hoveredCell.sampleIdx + 1}): {hoveredCell.value > 0 ? `+${hoveredCell.value.toFixed(2)}` : hoveredCell.value.toFixed(2)}
                  </span>
                )}
              </div>

              {/* Real Heatmap Grid Visualization */}
              {isLoadingHeatmap ? (
                <div className="h-40 flex items-center justify-center text-xs text-slate-400">
                  <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading standardized z-scores...
                </div>
              ) : heatmapData && heatmapData.symbols.length > 0 ? (
                <div className="flex gap-2">
                  <div className="flex-1 space-y-1">
                    {heatmapData.symbols.slice(0, 8).map((symbol, rowIdx) => {
                      const values = heatmapData.z_scores[rowIdx] || [];
                      return (
                        <div key={symbol} className="flex items-center gap-1.5">
                          <span className="w-16 text-[10px] font-mono font-medium text-slate-600 truncate text-right">
                            {symbol}
                          </span>
                          <div className="flex-1 grid grid-cols-12 gap-1">
                            {values.slice(0, 12).map((v, colIdx) => {
                              let cellBg = '#F1F5F9';
                              if (v > 0) {
                                const intensity = Math.min(v / 2.0, 1.0);
                                cellBg = intensity > 0.65 ? '#EF4444' : intensity > 0.35 ? '#F87171' : '#FCA5A5';
                              } else if (v < 0) {
                                const intensity = Math.min(Math.abs(v) / 2.0, 1.0);
                                cellBg = intensity > 0.65 ? '#3B82F6' : intensity > 0.35 ? '#60A5FA' : '#93C5FD';
                              }
                              const isHovered = hoveredCell?.name === symbol && hoveredCell?.sampleIdx === colIdx;

                              return (
                                <div
                                  key={colIdx}
                                  onMouseEnter={() => setHoveredCell({ name: symbol, sampleIdx: colIdx, value: v })}
                                  onMouseLeave={() => setHoveredCell(null)}
                                  className={`h-4.5 rounded-[3px] transition-all duration-150 cursor-pointer ${
                                    isHovered ? 'ring-2 ring-slate-900 scale-110 z-10' : 'hover:opacity-90'
                                  }`}
                                  style={{ backgroundColor: cellBg }}
                                  title={`${symbol} [Sample ${colIdx + 1}]: ${v > 0 ? '+' : ''}${v.toFixed(2)} z-score`}
                                />
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Color Legend Scale */}
                  <div className="flex flex-col items-center justify-between text-[9px] font-mono text-slate-400 py-0.5 pl-1.5 shrink-0">
                    <span className="text-red-500 font-semibold">+2</span>
                    <div className="w-2 flex-1 my-1 rounded-full bg-gradient-to-b from-[#EF4444] via-[#F1F5F9] to-[#3B82F6] border border-slate-200 shadow-2xs" />
                    <span className="text-slate-400 font-semibold">0</span>
                    <div className="w-2 flex-1 my-1 rounded-full bg-gradient-to-b from-[#F1F5F9] to-[#3B82F6] border border-slate-200 shadow-2xs" />
                    <span className="text-blue-500 font-semibold">-2</span>
                  </div>
                </div>
              ) : (
                <div className="py-8 text-center text-xs text-slate-400">
                  No heatmap data available.
                </div>
              )}
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>Standardized z-scores across cohort</span>
              <Link to="/data-explorer" className="text-blue-600 font-semibold hover:underline">
                Explore matrix →
              </Link>
            </div>
          </div>

          {/* 3. Key Biomarkers Card (Real Integrated Gradients) */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-bold text-slate-900 text-base">Key Biomarkers</h3>
                <span className="text-[10px] text-slate-400 font-mono">Integrated Gradients</span>
              </div>

              {/* Biomarkers Table */}
              <div className="w-full text-left">
                <div className="grid grid-cols-12 pb-2 border-b border-slate-100 text-[10.5px] font-semibold text-slate-400 uppercase tracking-wider">
                  <div className="col-span-4">Feature</div>
                  <div className="col-span-5">Attribution</div>
                  <div className="col-span-3 text-right">Direction</div>
                </div>

                {isLoadingBiomarkers ? (
                  <div className="py-8 text-center text-xs text-slate-400">
                    <RefreshCw className="w-4 h-4 animate-spin mx-auto mb-2" />
                    Computing Integrated Gradients...
                  </div>
                ) : biomarkers.length > 0 ? (
                  <div className="divide-y divide-slate-50 mt-1">
                    {biomarkers.slice(0, 5).map((bm) => {
                      const maxImp = Math.max(...biomarkers.map((b) => b.importance || 0.001));
                      const normWidth = Math.min((bm.importance / maxImp) * 100, 100);
                      return (
                        <div key={bm.gene} className="grid grid-cols-12 py-2 items-center text-xs">
                          <div className="col-span-4 font-mono font-bold text-slate-800 text-[11.5px] truncate" title={bm.gene}>
                            {bm.gene}
                          </div>
                          <div className="col-span-5 pr-2 flex items-center gap-2">
                            <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
                              <div
                                className="h-full rounded-full bg-blue-500"
                                style={{ width: `${normWidth}%` }}
                              />
                            </div>
                            <span className="font-mono text-[10.5px] text-slate-600">
                              {bm.importance.toFixed(3)}
                            </span>
                          </div>
                          <div className="col-span-3 text-right">
                            {bm.effect === 'positive' ? (
                              <span className="inline-flex items-center gap-0.5 text-emerald-600 font-medium text-[11px]">
                                <ArrowUpRight className="w-3.5 h-3.5" />
                                Pos
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-0.5 text-rose-500 font-medium text-[11px]">
                                <ArrowDownRight className="w-3.5 h-3.5" />
                                Neg
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="py-8 text-center text-xs text-slate-400">
                    No biomarkers loaded for this patient.
                  </div>
                )}
              </div>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>Top causal drivers (50 steps)</span>
              <Link to="/biomarkers" className="text-blue-600 font-semibold hover:underline">
                View all biomarkers →
              </Link>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* BOTTOM SECTION (3 Columns): Cohort Analysis | Model Insights | Pathway Analysis */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 1. Cohort Analysis Card (Real Distribution & Real Embeddings) */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            <div>
              <div className="mb-2">
                <h3 className="font-bold text-slate-900 text-base">Cohort Analysis</h3>
                <p className="text-[11px] text-slate-400">
                  Explore overall dataset distribution and 2D embedding space across PAM50 subtypes.
                </p>
              </div>

              {/* Tabs */}
              <div className="flex items-center gap-1 p-1 bg-slate-50 rounded-xl mb-4 border border-slate-100">
                <button
                  onClick={() => setCohortTab('distribution')}
                  className={`flex-1 py-1 px-2 rounded-lg text-[10.5px] font-semibold transition-all ${
                    cohortTab === 'distribution'
                      ? 'bg-[#0B1B3F] text-white shadow-2xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  Subtype Distribution
                </button>
                <button
                  onClick={() => setCohortTab('tsne')}
                  className={`flex-1 py-1 px-2 rounded-lg text-[10.5px] font-semibold transition-all ${
                    cohortTab === 'tsne'
                      ? 'bg-[#0B1B3F] text-white shadow-2xs'
                      : 'text-slate-500 hover:text-slate-800'
                  }`}
                >
                  UMAP Projection
                </button>
              </div>

              {cohortTab === 'distribution' ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-700 mb-1">
                    PAM50 Subtype Distribution (n = {cohortStats?.total_patients ?? 646})
                  </div>
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={subtypeDistData} margin={{ top: 15, right: 5, bottom: 0, left: -25 }}>
                        <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748B' }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 9, fill: '#94A3B8' }} axisLine={false} tickLine={false} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#FFFFFF', borderRadius: '12px', border: '1px solid #E2E8F0', fontSize: '11px' }}
                        />
                        <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                          {subtypeDistData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div>
                  <div className="text-[11px] font-semibold text-slate-700 mb-1 flex items-center justify-between">
                    <span>2D Embedding Space (Fused)</span>
                    <span className="text-[10px] text-slate-400 font-mono">646 patients</span>
                  </div>
                  <div className="h-44 w-full bg-slate-50/60 rounded-xl border border-slate-100 p-2 relative flex flex-col justify-between">
                    {scatterBounds ? (
                      <svg className="w-full h-32" viewBox="0 0 100 100">
                        {embeddingsPoints.map((pt) => {
                          const cx = ((pt.x - scatterBounds.minX) / scatterBounds.spanX) * 90 + 5;
                          const cy = ((pt.y - scatterBounds.minY) / scatterBounds.spanY) * 90 + 5;
                          const isSelected = pt.patient_id === selectedPatientId;
                          return (
                            <circle
                              key={pt.patient_id}
                              cx={cx}
                              cy={cy}
                              r={isSelected ? 4 : 2}
                              fill={SUBTYPE_COLORS[pt.true_pam50] || '#94A3B8'}
                              opacity={isSelected ? 1 : 0.75}
                              stroke={isSelected ? '#0F172A' : 'none'}
                              strokeWidth={isSelected ? 1.5 : 0}
                            >
                              <title>{`${pt.patient_id} (${pt.true_pam50})`}</title>
                            </circle>
                          );
                        })}
                      </svg>
                    ) : (
                      <div className="h-32 flex items-center justify-center text-xs text-slate-400">
                        Loading embeddings...
                      </div>
                    )}
                    <div className="flex flex-wrap items-center justify-center gap-2 pt-1 border-t border-slate-100 text-[9px] text-slate-500 font-medium">
                      {Object.entries(SUBTYPE_COLORS).map(([st, col]) => (
                        <span key={st} className="flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: col }} />
                          {st}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>TCGA-BRCA 5-fold CV</span>
              <Link to="/cohort-analysis" className="text-blue-600 font-semibold hover:underline">
                Full cohort view →
              </Link>
            </div>
          </div>

          {/* 2. Model Insights Card (Real Ego-Graph Visual) */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between">
            <div>
              <div className="mb-2">
                <h3 className="font-bold text-slate-900 text-base">Model Insights</h3>
              </div>

              {/* Tabs */}
              <div className="flex items-center gap-1.5 p-1 bg-slate-50 rounded-xl mb-3 border border-slate-100">
                <button
                  className="flex-1 py-1 px-2 rounded-lg text-[11px] font-semibold bg-[#0B1B3F] text-white shadow-2xs"
                >
                  Patient Similarity Graph
                </button>
              </div>

              <div className="text-[11px] font-semibold text-slate-700 mb-1">
                Ego Network for {selectedPatientId}
              </div>

              {/* Real Patient Similarity Graph Visual */}
              <div className="h-44 w-full bg-slate-50/50 rounded-xl border border-slate-100 p-2 relative flex flex-col justify-between">
                {egoGraph && egoGraph.nodes.length > 0 ? (
                  <svg className="w-full h-36" viewBox="0 0 240 140">
                    {/* Render real edges */}
                    <g stroke="#CBD5E1" strokeWidth="0.8" opacity="0.6">
                      {egoGraph.edges.slice(0, 20).map((edge, idx) => {
                        const targetIdx = egoGraph.nodes.findIndex((n) => n.patient_id === edge.target);
                        if (targetIdx <= 0) return null;
                        // Visual-only layout geometry (polar coordinates for SVG circular layout)
                        const angle = ((targetIdx - 1) / (egoGraph.nodes.length - 1)) * 2 * Math.PI;
                        const targetX = 120 + 75 * Math.cos(angle);
                        const targetY = 70 + 45 * Math.sin(angle);
                        return (
                          <line
                            key={idx}
                            x1={120}
                            y1={70}
                            x2={targetX}
                            y2={targetY}
                            stroke={edge.directed_knn_flag ? '#94A3B8' : '#CBD5E1'}
                            strokeDasharray={edge.directed_knn_flag ? 'none' : '3,3'}
                          />
                        );
                      })}
                    </g>

                    {/* Neighbor Nodes */}
                    {egoGraph.nodes.slice(1).map((node, idx) => {
                      const angle = (idx / (egoGraph.nodes.length - 1)) * 2 * Math.PI;
                      const nx = 120 + 75 * Math.cos(angle);
                      const ny = 70 + 45 * Math.sin(angle);
                      const nodeColor = SUBTYPE_COLORS[node.pam50 || 'Normal'] || '#94A3B8';
                      return (
                        <g key={node.patient_id}>
                          <circle
                            cx={nx}
                            cy={ny}
                            r={4.5}
                            fill={nodeColor}
                          >
                            <title>{`${node.patient_id} (${node.pam50 || 'Unknown'})`}</title>
                          </circle>
                        </g>
                      );
                    })}

                    {/* Center Node */}
                    <circle cx={120} cy={70} r={6.5} fill={theme.hex} />
                    <circle cx={120} cy={70} r={9.5} fill="none" stroke={theme.hex} strokeWidth={1.5} strokeDasharray="3,3" />
                  </svg>
                ) : (
                  <div className="h-36 flex items-center justify-center text-xs text-slate-400">
                    Loading ego graph...
                  </div>
                )}

                {/* Subtype Legend Right Overlay */}
                <div className="absolute right-2 top-2 bg-white/90 backdrop-blur-xs p-1.5 rounded-lg border border-slate-200/60 shadow-2xs space-y-1 text-[9.5px]">
                  {Object.entries(SUBTYPE_COLORS).map(([st, col]) => (
                    <div key={st} className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: col }} />
                      <span className="text-slate-600">{st}</span>
                    </div>
                  ))}
                </div>
              </div>

              <p className="text-[10px] text-slate-400 mt-2 leading-relaxed">
                Center node is {selectedPatientId}. Edges connect to its 10 nearest neighbors in cosine-similarity space (k=10, symmetrized). Layout positions have no biological meaning.
              </p>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>Cosine kNN (k=10)</span>
              <Link to="/graph-explorer" className="text-blue-600 font-semibold hover:underline">
                Explore graph →
              </Link>
            </div>
          </div>

          {/* 3. Pathway Analysis Card (Planned / Disabled state per R1 & R3) */}
          <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] flex flex-col justify-between opacity-90">
            <div>
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-slate-900 text-base">Pathway Analysis</h3>
                <span className="px-2 py-0.5 rounded-full bg-purple-50 border border-purple-200 text-purple-700 text-[10px] font-bold">
                  Planned
                </span>
              </div>

              <div className="mt-4 p-4 rounded-xl bg-purple-50/50 border border-purple-100 space-y-3">
                <div className="flex items-start gap-2.5">
                  <div className="w-7 h-7 rounded-lg bg-purple-100 text-purple-600 flex items-center justify-center shrink-0 mt-0.5">
                    <Lightbulb className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-slate-900">External Gene Sets Required</h4>
                    <p className="text-[11px] text-slate-600 mt-1 leading-relaxed">
                      Pathway enrichment requires curated MSigDB Hallmark or KEGG <code className="text-purple-700 font-mono">.gmt</code> database files.
                      In adherence to zero-mock rules, synthetic pathway scores are disabled.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-4 space-y-2 text-[11px] text-slate-500">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                  <span>Hypergeometric Over-Representation Test</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                  <span>Benjamini-Hochberg FDR Correction</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                  <span>Subtype-Specific Pathway Mapping</span>
                </div>
              </div>
            </div>

            <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
              <span>Status: Planned for v2</span>
              <Link to="/pathway-analysis" className="text-purple-600 font-semibold hover:underline">
                View roadmap →
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
