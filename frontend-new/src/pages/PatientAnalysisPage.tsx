import { useState, useEffect, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard, TabPills, ProgressBar, ErrorState, ResearchDisclaimer, GlossaryTerm } from '@/components/ui';
import {
  SUBTYPE_COLORS, SUBTYPES, MODALITY_COLORS, MODALITY_LABELS,
  type PAM50Subtype, type Patient, type PredictionResult,
  type ExplanationResult, type OmicsProfile, type CausalValidation,
} from '@/lib/types';
import { api, ApiError } from '@/api/client';
import { useHealth } from '@/lib/health';
import {
  Search, Check, ArrowRight, User, Target, BarChart2,
  ExternalLink, Loader2, Sparkles, ChevronDown, Shuffle,
  Upload, AlertTriangle, X, Play, RefreshCw, Layers, ShieldAlert,
} from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { PatientGraphExplorer } from '@/components/PatientGraphExplorer';

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

const STEPPER_ITEMS = [
  { id: 'info', label: '1. Patient Information', sublabel: 'Clinical fields & history' },
  { id: 'omics', label: '2. Multi-Omics Profiles', sublabel: '5,000 / 1,881 / 3,957 features' },
  { id: 'prediction', label: '3. Prediction Result', sublabel: 'Out-of-fold PAM50' },
  { id: 'interpretation', label: '4. Interpretation & Ablation', sublabel: 'IG attributions & live causal test' },
  { id: 'graph', label: '5. Patient Graph', sublabel: 'kNN ego neighborhood' },
];

export function PatientAnalysisPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialPid = searchParams.get('patient') || 'TCGA-AR-A1AR';

  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string>(initialPid);
  const [activeStep, setActiveStep] = useState<string>('info');
  const [showDropdown, setShowDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [explainLayer, setExplainLayer] = useState<'rna' | 'mirna' | 'meth'>('rna');
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);

  // Clinical data for selected patient
  const [clinicalData, setClinicalData] = useState<Record<string, unknown> | null>(null);

  // New Patient Upload Modal State
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  const health = useHealth();
  const live = health.mode === 'live';

  const loadPatients = () => {
    setLoadError(null);
    api.getPatients()
      .then(pts => {
        setPatients(pts);
        if (pts.length > 0 && !pts.some(p => p.id === selectedPatient)) {
          setSelectedPatient(pts[0].id);
        }
      })
      .catch(err => setLoadError(err.message || 'Failed to load patients'));
  };

  useEffect(() => {
    loadPatients();
  }, []);

  // Update selected patient when query param changes
  useEffect(() => {
    const p = searchParams.get('patient');
    if (p && p !== selectedPatient) {
      setSelectedPatient(p);
    }
  }, [searchParams]);

  // Load clinical data and out-of-fold prediction when selected patient changes
  useEffect(() => {
    if (!selectedPatient) return;
    setRunError(null);
    setPrediction(null);
    setExplanation(null);

    // Fetch clinical info
    api.getPatientClinical(selectedPatient)
      .then(setClinicalData)
      .catch(() => setClinicalData(null));

    // Fetch prediction
    api.predict(selectedPatient)
      .then(setPrediction)
      .catch((err) => {
        console.error('Prediction load error:', err);
      });
  }, [selectedPatient]);

  // Fetch explanation when on interpretation step
  useEffect(() => {
    if (!selectedPatient) return;
    const ac = new AbortController();
    api.getExplanation(selectedPatient, 20, explainLayer, ac.signal)
      .then(setExplanation)
      .catch(() => setExplanation(null));
    return () => ac.abort();
  }, [explainLayer, selectedPatient]);

  const handleRunPrediction = async () => {
    if (!selectedPatient) return;
    setIsRunning(true);
    setRunError(null);
    try {
      const res = await api.predict(selectedPatient);
      setPrediction(res);
      setActiveStep('prediction');
    } catch (err: any) {
      setRunError(err.message || 'Prediction failed');
    } finally {
      setIsRunning(false);
    }
  };

  const filteredPatients = patients.filter(p =>
    p.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (p.subtype && p.subtype.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const selectedMeta = patients.find(p => p.id === selectedPatient);

  return (
    <div>
      <TopBar
        title="Patient Analysis"
        subtitle="Multi-omics profiling, out-of-fold PAM50 prediction, and causal ablation analysis."
        onSelectPatient={(pid) => {
          setSelectedPatient(pid);
          setSearchParams({ patient: pid });
        }}
      />

      <div className="p-6">
        {/* Header Action: New Patient Upload */}
        <div className="flex justify-between items-center mb-5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Cohort ID:</span>
            <span className="text-base font-bold font-mono text-slate-900">{selectedPatient}</span>
            {selectedMeta?.true_pam50 && (
              <span
                className="px-2 py-0.5 rounded text-xs font-bold"
                style={{
                  backgroundColor: `${SUBTYPE_COLORS[selectedMeta.true_pam50]}20`,
                  color: SUBTYPE_COLORS[selectedMeta.true_pam50],
                }}
              >
                True: {selectedMeta.true_pam50}
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={() => setIsUploadModalOpen(true)}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-[#0B1B3F] hover:bg-[#1A2D5A] text-white text-xs font-semibold shadow-2xs transition-colors"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload New Patient (Experimental)</span>
          </button>
        </div>

        {loadError && <ErrorState error={loadError} onRetry={loadPatients} />}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column (Navigation & Stepper) */}
          <div className="lg:col-span-4 space-y-4">
            {/* Patient Selector Card */}
            <div className="bio-card p-5">
              <label className="text-xs font-semibold text-slate-500 block mb-1.5">Select Cohort Patient</label>
              <div className="relative">
                <button
                  type="button"
                  onClick={() => setShowDropdown(!showDropdown)}
                  className="w-full flex items-center justify-between px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs font-mono font-medium text-slate-800 hover:border-blue-500 transition-colors"
                >
                  <span>{selectedPatient || 'Select patient...'}</span>
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                </button>

                {showDropdown && (
                  <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-slate-200 rounded-xl shadow-xl z-30 max-h-64 overflow-y-auto p-1.5">
                    <div className="p-1 border-b border-slate-100 mb-1">
                      <div className="flex items-center gap-2 px-2 py-1.5 bg-slate-50 rounded-lg">
                        <Search className="w-3.5 h-3.5 text-slate-400" />
                        <input
                          autoFocus
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          placeholder="Search 646 cohort..."
                          className="flex-1 text-xs outline-none bg-transparent text-slate-800"
                        />
                      </div>
                    </div>
                    {filteredPatients.map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => {
                          setSelectedPatient(p.id);
                          setSearchParams({ patient: p.id });
                          setShowDropdown(false);
                          setSearchQuery('');
                        }}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs transition-colors ${
                          p.id === selectedPatient ? 'bg-blue-50 text-blue-700 font-bold' : 'hover:bg-slate-50 text-slate-700'
                        }`}
                      >
                        <span className="font-mono">{p.id}</span>
                        {p.subtype && (
                          <span
                            className="px-1.5 py-0.5 rounded text-[10px] font-semibold"
                            style={{
                              backgroundColor: `${SUBTYPE_COLORS[p.subtype]}15`,
                              color: SUBTYPE_COLORS[p.subtype],
                            }}
                          >
                            {p.subtype}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={handleRunPrediction}
                disabled={isRunning}
                className="w-full mt-3 inline-flex items-center justify-center gap-2 py-2.5 rounded-xl bg-[#0B1B3F] hover:bg-[#1A2D5A] disabled:opacity-50 text-white text-xs font-semibold transition-colors shadow-2xs"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Loading Out-Of-Fold Result...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Run Out-Of-Fold Prediction</span>
                  </>
                )}
              </button>
            </div>

            {/* Stepper Card */}
            <div className="bio-card p-5">
              <div className="space-y-1">
                {STEPPER_ITEMS.map((item, i) => {
                  const isActive = activeStep === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => setActiveStep(item.id)}
                      className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all text-left ${
                        isActive ? 'bg-blue-50/70 border border-blue-200/60 shadow-2xs' : 'hover:bg-slate-50'
                      }`}
                    >
                      <div
                        className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold shrink-0 ${
                          isActive ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-500'
                        }`}
                      >
                        {i + 1}
                      </div>
                      <div>
                        <p className={`text-xs font-bold ${isActive ? 'text-blue-900' : 'text-slate-800'}`}>
                          {item.label}
                        </p>
                        <p className="text-[11px] text-slate-400 mt-0.5">{item.sublabel}</p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column (Step Content) */}
          <div className="lg:col-span-8 space-y-4">
            {runError && <ErrorState error={runError} onRetry={handleRunPrediction} />}

            {/* STEP 1: Patient Information */}
            {activeStep === 'info' && (
              <PatientInfoStep
                patientId={selectedPatient}
                clinical={clinicalData}
                meta={selectedMeta}
                prediction={prediction}
              />
            )}

            {/* STEP 2: Multi-Omics Profiles */}
            {activeStep === 'omics' && (
              <OmicsProfileStep patientId={selectedPatient} />
            )}

            {/* STEP 3: Prediction Result */}
            {activeStep === 'prediction' && (
              <PredictionResultStep prediction={prediction} />
            )}

            {/* STEP 4: Interpretation & Live Causal Ablation */}
            {activeStep === 'interpretation' && (
              <InterpretationStep
                patientId={selectedPatient}
                prediction={prediction}
                explanation={explanation}
                layer={explainLayer}
                onLayerChange={setExplainLayer}
              />
            )}

            {/* STEP 5: Patient Graph Explorer */}
            {activeStep === 'graph' && (
              <PatientGraphExplorer
                initialPatientId={selectedPatient}
                onSelectPatient={(pid) => {
                  setSelectedPatient(pid);
                  setSearchParams({ patient: pid });
                }}
              />
            )}
          </div>
        </div>
      </div>

      {/* Upload Modal (Inductive Inference) */}
      <AnimatePresence>
        {isUploadModalOpen && (
          <UploadPatientModal onClose={() => setIsUploadModalOpen(false)} />
        )}
      </AnimatePresence>
    </div>
  );
}

// =========================================================================
// STEP 1: PATIENT INFORMATION
// =========================================================================

function PatientInfoStep({
  patientId,
  clinical,
  meta,
  prediction,
}: {
  patientId: string;
  clinical: Record<string, unknown> | null;
  meta?: Patient;
  prediction: PredictionResult | null;
}) {
  // Computed consistency check chip from real data (Rule R1: never invented)
  const consistencyNote = useMemo(() => {
    if (!clinical) return null;
    const er = String(clinical.er_status || '').toLowerCase();
    const her2 = String(clinical.her2_status || '').toLowerCase();
    const predicted = prediction?.predicted_subtype || meta?.true_pam50;

    if (er.includes('negative') && predicted === 'Basal') {
      return { text: 'ER-negative, consistent with Basal-like subtype', type: 'match' };
    }
    if (er.includes('positive') && (predicted === 'LumA' || predicted === 'LumB')) {
      return { text: `ER-positive, consistent with ${predicted} subtype`, type: 'match' };
    }
    if (her2.includes('positive') && predicted === 'Her2') {
      return { text: 'HER2-positive, consistent with HER2-enriched subtype', type: 'match' };
    }
    return null;
  }, [clinical, prediction, meta]);

  const fields = [
    { label: 'Age at Diagnosis', value: clinical?.age != null ? `${clinical.age} years` : 'Not available' },
    { label: 'Estrogen Receptor (ER)', value: clinical?.er_status != null ? String(clinical.er_status) : 'Not available' },
    { label: 'Progesterone Receptor (PR)', value: clinical?.pr_status != null ? String(clinical.pr_status) : 'Not available' },
    { label: 'HER2 IHC Status', value: clinical?.her2_status != null ? String(clinical.her2_status) : 'Not available' },
    { label: 'Histologic Grade', value: clinical?.neoplasm_histologic_grade != null ? String(clinical.neoplasm_histologic_grade) : 'Not available' },
    { label: 'Positive Lymph Nodes', value: clinical?.lymph_nodes_examined_positive != null ? String(clinical.lymph_nodes_examined_positive) : 'Not available' },
    { label: 'Histologic Type', value: clinical?.histological_type != null ? String(clinical.histological_type) : 'Not available' },
    { label: 'Overall Survival (Months)', value: clinical?.overall_survival_months != null ? `${Number(clinical.overall_survival_months).toFixed(1)} mo` : 'Not available' },
    { label: 'Vital Status', value: clinical?.vital_status != null ? String(clinical.vital_status) : 'Not available' },
  ];

  return (
    <SectionCard title="Patient Clinical Summary" subtitle="Demographics and histology from TCGA-BRCA PanCancer Atlas">
      <div className="space-y-4">
        {consistencyNote && (
          <div className="p-3 bg-emerald-50 border border-emerald-200/80 rounded-xl flex items-center gap-2 text-xs font-medium text-emerald-800">
            <Check className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>{consistencyNote.text}</span>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {fields.map((f) => (
            <div key={f.label} className="p-3 bg-slate-50 border border-slate-100 rounded-xl">
              <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400 block mb-0.5">
                {f.label}
              </span>
              <span className="text-xs font-bold text-slate-800">{f.value}</span>
            </div>
          ))}
        </div>

        <div className="p-3 bg-blue-50/60 border border-blue-100 rounded-xl text-xs text-blue-900 leading-relaxed">
          <strong>Validation Fold:</strong> Patient <code className="font-mono">{patientId}</code> was evaluated in fold{' '}
          <strong>{prediction?.fold_id ?? meta?.fold_id ?? 1}</strong> of the 5-fold cross-validation scheme.
          All three omics layers (RNA-seq, miRNA, and DNA methylation) are present in the locked cohort.
        </div>
      </div>
    </SectionCard>
  );
}

// =========================================================================
// STEP 2: MULTI-OMICS PROFILES
// =========================================================================

function OmicsProfileStep({ patientId }: { patientId: string }) {
  const [activeLayer, setActiveLayer] = useState<'rna' | 'mirna' | 'meth'>('rna');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<OmicsProfile | null>(null);
  const [filterQuery, setFilterQuery] = useState('');

  useEffect(() => {
    if (!patientId) return;
    setLoading(true);
    setError(null);
    api.getProfile(patientId, activeLayer, 40)
      .then(setProfile)
      .catch((err) => setError(err.message || 'Failed to load profile'))
      .finally(() => setLoading(false));
  }, [patientId, activeLayer]);

  const layerCounts = {
    rna: 5000,
    mirna: 1881,
    meth: 3957,
  };

  const filteredFeatures = (profile?.features || []).filter(
    (f) =>
      f.symbol.toLowerCase().includes(filterQuery.toLowerCase()) ||
      f.feature_id.toLowerCase().includes(filterQuery.toLowerCase())
  );

  const maxAbsVal = Math.max(0.1, ...(profile?.features || []).map((f) => Math.abs(f.value)));

  return (
    <SectionCard
      title="Multi-Omics Profile"
      subtitle={`Processed feature values for ${patientId} (Cohort size: ${layerCounts[activeLayer].toLocaleString()} features)`}
    >
      <div className="space-y-4">
        {/* Layer Tabs */}
        <div className="flex items-center gap-2 p-1 bg-slate-50 border border-slate-200 rounded-xl">
          <button
            type="button"
            onClick={() => setActiveLayer('rna')}
            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-bold transition-all ${
              activeLayer === 'rna' ? 'bg-[#0B1B3F] text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            RNA-seq (5,000 genes)
          </button>
          <button
            type="button"
            onClick={() => setActiveLayer('mirna')}
            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-bold transition-all ${
              activeLayer === 'mirna' ? 'bg-[#0B1B3F] text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            miRNA (1,881 features)
          </button>
          <button
            type="button"
            onClick={() => setActiveLayer('meth')}
            className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-bold transition-all ${
              activeLayer === 'meth' ? 'bg-[#0B1B3F] text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            DNA Methylation (3,957 probes)
          </button>
        </div>

        {/* Feature Search */}
        <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl">
          <Search className="w-4 h-4 text-slate-400 shrink-0" />
          <input
            type="text"
            placeholder="Search features by symbol or ID..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="flex-1 text-xs bg-transparent outline-none text-slate-800"
          />
        </div>

        {error && <ErrorState error={error} />}

        {loading ? (
          <div className="h-64 flex items-center justify-center text-xs text-slate-400">
            <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Loading processed features...
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-100">
                  <th className="text-left pb-2">Feature / Symbol</th>
                  <th className="text-left pb-2">Feature ID</th>
                  <th className="text-left pb-2 w-48">Standardized Magnitude</th>
                  <th className="text-right pb-2">Value</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {filteredFeatures.map((f) => {
                  const width = (Math.abs(f.value) / maxAbsVal) * 100;
                  return (
                    <tr key={f.feature_id} className="hover:bg-slate-50/70">
                      <td className="py-2 font-mono font-bold text-slate-900">{f.symbol}</td>
                      <td className="py-2 font-mono text-slate-500">{f.feature_id}</td>
                      <td className="py-2 pr-4">
                        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${f.value >= 0 ? 'bg-red-500' : 'bg-blue-500'}`}
                            style={{ width: `${Math.min(width, 100)}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-2 text-right font-mono font-bold">
                        <span className={f.value >= 0 ? 'text-red-600' : 'text-blue-600'}>
                          {f.value > 0 ? `+${f.value.toFixed(3)}` : f.value.toFixed(3)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </SectionCard>
  );
}

// =========================================================================
// STEP 3: PREDICTION RESULT
// =========================================================================

function PredictionResultStep({ prediction }: { prediction: PredictionResult | null }) {
  if (!prediction) {
    return (
      <SectionCard title="Prediction Result">
        <div className="py-12 text-center text-slate-400 text-xs">
          No out-of-fold prediction loaded yet. Click "Run Out-Of-Fold Prediction".
        </div>
      </SectionCard>
    );
  }

  const subtype = prediction.predicted_subtype;
  const theme = SUBTYPE_THEME[subtype] || SUBTYPE_THEME.LumA;

  const modalityData = [
    { name: 'RNA-seq', value: prediction.modality_weights.rna, fill: '#3B82F6' },
    { name: 'miRNA', value: prediction.modality_weights.mirna, fill: '#EC4899' },
    { name: 'DNA Methylation', value: prediction.modality_weights.methylation, fill: '#10B981' },
  ];

  return (
    <div className="space-y-4">
      <SectionCard
        title="Out-Of-Fold PAM50 Prediction"
        badge={
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-bold">
            Fold {prediction.fold_id} Validation
          </span>
        }
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="space-y-3">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Predicted Molecular Subtype
            </span>
            <div className="flex items-baseline gap-3">
              <span className={`text-4xl font-extrabold ${theme.text}`}>{subtype}</span>
              <span className="text-xs text-slate-400 font-medium">Confidence:</span>
              <span className="text-base font-bold text-slate-800">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>

            <div className="w-full h-3 rounded-full bg-slate-100 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.round(prediction.confidence * 100)}%`,
                  backgroundColor: theme.hex,
                }}
              />
            </div>

            {prediction.true_pam50 && (
              <div className="flex items-center gap-2 text-xs pt-1">
                <span className="text-slate-500 font-medium">True Subtype:</span>
                <span className="font-bold text-slate-800">{prediction.true_pam50}</span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    prediction.match ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                  }`}
                >
                  {prediction.match ? 'Match' : 'Mismatch'}
                </span>
              </div>
            )}

            <p className="text-xs text-slate-500 leading-relaxed mt-2">{theme.desc}</p>
          </div>

          {/* Probabilities Breakdown */}
          <div className="space-y-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Subtype Probabilities (Sum = 1.00)
            </span>
            {(['LumA', 'LumB', 'Her2', 'Basal', 'Normal'] as PAM50Subtype[]).map((name) => {
              const prob = prediction.probabilities?.[name] ?? 0;
              return (
                <div key={name} className="flex items-center gap-2 text-xs">
                  <span className="w-14 font-semibold text-slate-700">{name}</span>
                  <div className="flex-1 h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-300"
                      style={{
                        width: `${Math.max(prob * 100, 2)}%`,
                        backgroundColor: SUBTYPE_COLORS[name],
                      }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono text-[11px] font-bold text-slate-700">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </SectionCard>

      {/* Modality Fusion Contribution */}
      <SectionCard title="Modality Fusion Weights" subtitle="Learned softmax pooling weights across modalities">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
          <div className="h-44 w-full flex items-center justify-center">
            <ResponsiveContainer width={180} height={180}>
              <PieChart>
                <Pie
                  data={modalityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {modalityData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-3">
            <div className="space-y-2">
              {modalityData.map((m) => (
                <div key={m.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: m.fill }} />
                    <span className="font-semibold text-slate-800">{m.name}</span>
                  </div>
                  <span className="font-mono font-bold text-slate-900">{(m.value * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>

            <p className="text-xs text-slate-500 leading-relaxed border-t border-slate-100 pt-3">
              Generated caption: The fusion module assigned {(prediction.modality_weights.rna * 100).toFixed(0)}% weight to RNA,{' '}
              {(prediction.modality_weights.mirna * 100).toFixed(0)}% to miRNA, and{' '}
              {(prediction.modality_weights.methylation * 100).toFixed(0)}% to DNA methylation for patient {prediction.patient_id}.
            </p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// =========================================================================
// STEP 4: INTERPRETATION & LIVE CAUSAL ABLATION
// =========================================================================

function InterpretationStep({
  patientId,
  prediction,
  explanation,
  layer,
  onLayerChange,
}: {
  patientId: string;
  prediction: PredictionResult | null;
  explanation: ExplanationResult | null;
  layer: 'rna' | 'mirna' | 'meth';
  onLayerChange: (l: 'rna' | 'mirna' | 'meth') => void;
}) {
  const [topK, setTopK] = useState(10);
  const [selectedGenes, setSelectedGenes] = useState<string[]>([]);
  const [isAblating, setIsAblating] = useState(false);
  const [ablationResult, setAblationResult] = useState<Record<string, unknown> | null>(null);
  const [ablationError, setAblationError] = useState<string | null>(null);

  // Precomputed cohort causal validation
  const [causalStats, setCausalStats] = useState<CausalValidation | null>(null);

  useEffect(() => {
    api.getCausalValidation().then(setCausalStats).catch(console.error);
  }, []);

  const biomarkers = explanation?.biomarkers || [];
  const topBiomarkers = biomarkers.slice(0, topK);

  const handleToggleGene = (gene: string) => {
    setSelectedGenes((prev) =>
      prev.includes(gene) ? prev.filter((g) => g !== gene) : [...prev, gene]
    );
  };

  const handleSelectTopK = (k: number) => {
    const genes = biomarkers.slice(0, k).map((b) => b.gene);
    setSelectedGenes(genes);
  };

  const handleRunAblation = async (mode: 'custom' | 'top_k' | 'random_k' = 'custom') => {
    setIsAblating(true);
    setAblationError(null);
    try {
      const res = await api.ablateGenes({
        patient_id: patientId,
        layer,
        genes: mode === 'custom' ? selectedGenes : undefined,
        mode,
        k: mode !== 'custom' ? 5 : undefined,
      });
      setAblationResult(res);
    } catch (err: any) {
      setAblationError(err.message || 'Causal ablation failed');
    } finally {
      setIsAblating(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Integrated Gradients Attribution Card */}
      <SectionCard
        title="Integrated Gradients Explainability"
        subtitle={`Top attributed features for predicted class (${prediction?.predicted_subtype || 'Subtype'})`}
        actions={
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-lg">
              <button
                type="button"
                onClick={() => onLayerChange('rna')}
                className={`px-2 py-1 rounded text-xs font-semibold ${layer === 'rna' ? 'bg-white shadow-2xs text-blue-700' : 'text-slate-600'}`}
              >
                RNA
              </button>
              <button
                type="button"
                onClick={() => onLayerChange('mirna')}
                className={`px-2 py-1 rounded text-xs font-semibold ${layer === 'mirna' ? 'bg-white shadow-2xs text-pink-700' : 'text-slate-600'}`}
              >
                miRNA
              </button>
              <button
                type="button"
                onClick={() => onLayerChange('meth')}
                className={`px-2 py-1 rounded text-xs font-semibold ${layer === 'meth' ? 'bg-white shadow-2xs text-emerald-700' : 'text-slate-600'}`}
              >
                Methylation
              </button>
            </div>

            <label className="text-xs text-slate-500 flex items-center gap-1">
              <span>Top-K:</span>
              <input
                type="range"
                min={5}
                max={25}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-16 accent-blue-600"
              />
              <span className="font-bold text-slate-800 w-4">{topK}</span>
            </label>
          </div>
        }
      >
        <div className="space-y-4">
          {topBiomarkers.length === 0 ? (
            <div className="py-8 text-center text-xs text-slate-400">
              No Integrated Gradients attributions computed for this layer.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-100">
                    <th className="text-left pb-2 w-8">Ablate</th>
                    <th className="text-left pb-2">Gene / Symbol</th>
                    <th className="text-left pb-2">Feature ID</th>
                    <th className="text-left pb-2 w-48">Attribution Magnitude</th>
                    <th className="text-right pb-2">Effect</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {topBiomarkers.map((b) => (
                    <tr key={b.gene} className="hover:bg-slate-50/70">
                      <td className="py-2">
                        <input
                          type="checkbox"
                          checked={selectedGenes.includes(b.gene)}
                          onChange={() => handleToggleGene(b.gene)}
                          className="rounded text-blue-600 focus:ring-0"
                        />
                      </td>
                      <td className="py-2">
                        <a
                          href={`https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(b.gene)}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="font-mono font-bold text-blue-600 hover:underline inline-flex items-center gap-1"
                        >
                          {b.gene}
                          <ExternalLink className="w-3 h-3 text-slate-400" />
                        </a>
                      </td>
                      <td className="py-2 font-mono text-slate-400 text-[11px]">{b.feature_id}</td>
                      <td className="py-2 pr-4">
                        <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className="h-full rounded-full bg-blue-500"
                            style={{ width: `${Math.min(b.importance * 100, 100)}%` }}
                          />
                        </div>
                      </td>
                      <td className="py-2 text-right">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                            b.effect === 'positive' ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
                          }`}
                        >
                          {b.effect === 'positive' ? 'Positive' : 'Negative'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Live Causal Ablation Experiment Card */}
      <SectionCard
        title="Interactive Causal Ablation"
        subtitle="Zero out selected features and measure direct impact on the model's logits and probabilities"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-500 leading-relaxed">
            Select features from the table above or use quick buttons to run live inference forward passes with features zeroed.
            This demonstrates whether the model genuinely depends on these inputs.
          </p>

          <div className="flex flex-wrap items-center gap-2 pt-1">
            <button
              type="button"
              onClick={() => handleSelectTopK(5)}
              className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              Select Top 5
            </button>
            <button
              type="button"
              onClick={() => setSelectedGenes([])}
              className="px-3 py-1.5 rounded-lg border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              Clear Selection ({selectedGenes.length})
            </button>

            <button
              type="button"
              onClick={() => handleRunAblation('custom')}
              disabled={isAblating || selectedGenes.length === 0}
              className="ml-auto inline-flex items-center gap-1.5 px-4 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white text-xs font-semibold shadow-2xs"
            >
              {isAblating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3 h-3 fill-current" />}
              <span>Ablate Selected ({selectedGenes.length})</span>
            </button>

            <button
              type="button"
              onClick={() => handleRunAblation('random_k')}
              disabled={isAblating}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold"
            >
              <Shuffle className="w-3 h-3" />
              <span>Ablate Random 5</span>
            </button>
          </div>

          {ablationError && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
              {ablationError}
            </div>
          )}

          {/* Ablation Impact Results */}
          {ablationResult && (
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-900">
                  Ablation Result ({String(ablationResult.mode || 'custom')})
                </span>
                <span className="text-xs font-mono text-slate-500">
                  {String(ablationResult.n_features_ablated)} features ablated
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="p-2 bg-white rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-400 font-semibold block">Original Logit</span>
                  <span className="text-xs font-mono font-bold text-slate-800">
                    {Number(ablationResult.original_logit).toFixed(3)}
                  </span>
                </div>
                <div className="p-2 bg-white rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-400 font-semibold block">Ablated Logit</span>
                  <span className="text-xs font-mono font-bold text-slate-800">
                    {Number(ablationResult.ablated_logit).toFixed(3)}
                  </span>
                </div>
                <div className="p-2 bg-white rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-400 font-semibold block">Logit Shift (Δ)</span>
                  <span className="text-xs font-mono font-bold text-rose-600">
                    {Number(ablationResult.logit_delta) > 0 ? `+${Number(ablationResult.logit_delta).toFixed(3)}` : Number(ablationResult.logit_delta).toFixed(3)}
                  </span>
                </div>
                <div className="p-2 bg-white rounded-lg border border-slate-100">
                  <span className="text-[10px] text-slate-400 font-semibold block">Probability Drop</span>
                  <span className="text-xs font-mono font-bold text-rose-600">
                    -{(Number(ablationResult.probability_drop) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </SectionCard>

      {/* Cohort-Level Causal Validation Card (Rule R1: wording strictly from file) */}
      {causalStats && (
        <SectionCard title="Cohort Causal Validation" subtitle="50-patient systematic ablation test">
          <div className="space-y-3">
            <p className="text-xs text-slate-600 leading-relaxed">
              {causalStats.wording || (
                `Attribution-flagged genes changed the model's output more than random genes in ${causalStats.patients_larger_impact} of ${causalStats.total_patients} tested patients (Wilcoxon p=${causalStats.p_value}). This does not prove biological causation.`
              )}
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-center">
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Attributed Median Drop</span>
                <span className="text-lg font-bold text-blue-600 font-mono">
                  {causalStats.attributed_impact_median.toFixed(3)}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Random Median Drop</span>
                <span className="text-lg font-bold text-slate-500 font-mono">
                  {causalStats.random_impact_median.toFixed(3)}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">Impact Ratio</span>
                <span className="text-lg font-bold text-emerald-600 font-mono">
                  {causalStats.fold_difference.toFixed(1)}×
                </span>
              </div>
            </div>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

// =========================================================================
// EXPERIMENTAL UPLOAD MODAL (INDUCTIVE INFERENCE)
// =========================================================================

function UploadPatientModal({ onClose }: { onClose: () => void }) {
  const [rnaFile, setRnaFile] = useState<File | null>(null);
  const [mirnaFile, setMirnaFile] = useState<File | null>(null);
  const [methFile, setMethFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rnaFile && !mirnaFile && !methFile) {
      setError('Please provide at least one omics data file.');
      return;
    }

    setIsUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    if (rnaFile) formData.append('rna_file', rnaFile);
    if (mirnaFile) formData.append('mirna_file', mirnaFile);
    if (methFile) formData.append('meth_file', methFile);

    try {
      const res = await api.predictNewPatient(formData);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Upload inference failed.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-2xl border border-slate-200 p-6 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-base font-bold text-slate-900">Upload New Patient Data</h3>
            <p className="text-xs text-slate-500 mt-0.5">Experimental inductive inference via kNN graph attachment</p>
          </div>
          <button type="button" onClick={onClose} className="p-1 rounded-lg text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Warning Banner */}
        <div className="p-3 bg-amber-50 border border-amber-200 rounded-xl flex items-start gap-2.5 text-xs text-amber-900">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <strong className="block font-bold">Experimental, Not Validated:</strong>
            New patient inference dynamically links the sample to the 646-patient graph via cosine similarity.
            Uploaded files are processed in memory and immediately discarded.
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">RNA-seq (CSV / TSV, 5,000 features):</label>
            <input
              type="file"
              accept=".csv,.tsv,.txt"
              onChange={(e) => setRnaFile(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">miRNA (CSV / TSV, 1,881 features):</label>
            <input
              type="file"
              accept=".csv,.tsv,.txt"
              onChange={(e) => setMirnaFile(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-pink-50 file:text-pink-700 hover:file:bg-pink-100"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-700 block mb-1">DNA Methylation (CSV / TSV, 3,957 features):</label>
            <input
              type="file"
              accept=".csv,.tsv,.txt"
              onChange={(e) => setMethFile(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-600 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-emerald-50 file:text-emerald-700 hover:file:bg-emerald-100"
            />
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 text-red-700 text-xs rounded-xl">
              {error}
            </div>
          )}

          {result && (
            <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-xs space-y-2">
              <div className="font-bold text-emerald-900">Prediction Complete:</div>
              <div>Predicted Subtype: <strong className="text-emerald-800">{String(result.predicted_subtype)}</strong></div>
              <div>Confidence: <strong>{(Number(result.confidence) * 100).toFixed(1)}%</strong></div>
              <div className="text-[10px] text-emerald-700 font-mono">Prediction Type: {String(result.prediction_type)}</div>
            </div>
          )}

          <div className="pt-2 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 rounded-xl"
            >
              Close
            </button>
            <button
              type="submit"
              disabled={isUploading}
              className="px-4 py-2 text-xs font-semibold text-white bg-[#0B1B3F] hover:bg-[#1A2D5A] rounded-xl shadow-2xs disabled:opacity-50 inline-flex items-center gap-2"
            >
              {isUploading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Upload className="w-3.5 h-3.5" />}
              <span>Run Inductive Prediction</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
