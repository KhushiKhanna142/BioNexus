import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard } from '@/components/ui';
import { api } from '@/api/client';
import type { CaseQCReportResponse, QCModalityItem } from '@/lib/types';
import {
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Download,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  Info,
  HelpCircle,
} from 'lucide-react';

interface ModalityFileState {
  file: File | null;
  name: string;
  size: string;
  progress: number;
}

export function NewCasePage() {
  const navigate = useNavigate();

  // Three drag-and-drop slots
  const [files, setFiles] = useState<{
    rna: ModalityFileState;
    mirna: ModalityFileState;
    meth: ModalityFileState;
  }>({
    rna: { file: null, name: '', size: '', progress: 0 },
    mirna: { file: null, name: '', size: '', progress: 0 },
    meth: { file: null, name: '', size: '', progress: 0 },
  });

  const [patientRef, setPatientRef] = useState('');
  const [autoNormalize, setAutoNormalize] = useState(false);
  const [qcReport, setQcReport] = useState<CaseQCReportResponse | null>(null);
  const [checkingQC, setCheckingQC] = useState(false);
  const [runningPrediction, setRunningPrediction] = useState(false);
  const [isDemoData, setIsDemoData] = useState(false);
  const [showCautionModal, setShowCautionModal] = useState(false);

  // File input refs
  const rnaInputRef = useRef<HTMLInputElement>(null);
  const mirnaInputRef = useRef<HTMLInputElement>(null);
  const methInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  const handleFileSelect = (mod: 'rna' | 'mirna' | 'meth', file: File) => {
    setFiles((prev) => ({
      ...prev,
      [mod]: {
        file,
        name: file.name,
        size: formatFileSize(file.size),
        progress: 100,
      },
    }));
    setIsDemoData(false);
    triggerQC(file, mod);
  };

  const triggerQC = async (newFile?: File, newMod?: 'rna' | 'mirna' | 'meth') => {
    const fd = new FormData();
    if (newMod === 'rna' && newFile) fd.append('rna', newFile);
    else if (files.rna.file) fd.append('rna', files.rna.file);

    if (newMod === 'mirna' && newFile) fd.append('mirna', newFile);
    else if (files.mirna.file) fd.append('mirna', files.mirna.file);

    if (newMod === 'meth' && newFile) fd.append('meth', newFile);
    else if (files.meth.file) fd.append('meth', files.meth.file);

    fd.append('auto_normalize', autoNormalize ? 'true' : 'false');

    try {
      setCheckingQC(true);
      const res = await api.runQC(fd);
      setQcReport(res);
    } catch (err) {
      console.error('QC check failed:', err);
    } finally {
      setCheckingQC(false);
    }
  };

  // Load example case from real TCGA fixtures
  const handleLoadDemoCase = async () => {
    setCheckingQC(true);
    try {
      const demo = await api.getDemoCase();
      setIsDemoData(true);
      setPatientRef(String(demo.patient_ref || 'DEMO-TCGA-3C-AAAU'));

      // Create synthetic demo files for UI representation
      const createDummyFile = (name: string, count: number) =>
        new File([new Blob([`id,value\nfeature_1,2.5\n`])], name, { type: 'text/csv' });

      setFiles({
        rna: { file: createDummyFile('demo_rna_seq.csv', 5000), name: 'demo_rna_seq.csv', size: '142 KB', progress: 100 },
        mirna: { file: createDummyFile('demo_mirna.csv', 1881), name: 'demo_mirna.csv', size: '48 KB', progress: 100 },
        meth: { file: createDummyFile('demo_methylation.csv', 3957), name: 'demo_methylation.csv', size: '96 KB', progress: 100 },
      });

      // Synthetic demo QC report
      setQcReport({
        overall_status: 'PASSED',
        modalities_checked: ['rna', 'mirna', 'meth'],
        can_proceed: true,
        requires_caution: false,
        ready_modalities_count: 3,
        summary_message: 'Ready to run with 3 of 3 data types (TCGA-BRCA demo fixture).',
        auto_normalized_modalities: [],
        results: {
          rna: {
            modality: 'rna',
            status: 'PASSED',
            total_expected: 5000,
            mapped_count: 5000,
            coverage_pct: 1.0,
            missing_pct: 0.0,
            value_min: 0.0,
            value_max: 16.4,
            value_mean: 4.82,
            detected_scale: 'log2',
            likely_raw_counts: false,
            duplicate_ids_count: 0,
            duplicate_ids: [],
            warnings: [],
            errors: [],
            fix_hints: [],
            auto_normalized_applied: false,
          },
          mirna: {
            modality: 'mirna',
            status: 'PASSED',
            total_expected: 1881,
            mapped_count: 1881,
            coverage_pct: 1.0,
            missing_pct: 0.0,
            value_min: 0.0,
            value_max: 18.2,
            value_mean: 6.14,
            detected_scale: 'log2',
            likely_raw_counts: false,
            duplicate_ids_count: 0,
            duplicate_ids: [],
            warnings: [],
            errors: [],
            fix_hints: [],
            auto_normalized_applied: false,
          },
          meth: {
            modality: 'meth',
            status: 'PASSED',
            total_expected: 3957,
            mapped_count: 3957,
            coverage_pct: 1.0,
            missing_pct: 0.0,
            value_min: 0.02,
            value_max: 0.98,
            value_mean: 0.51,
            detected_scale: 'beta_values',
            likely_raw_counts: false,
            duplicate_ids_count: 0,
            duplicate_ids: [],
            warnings: [],
            errors: [],
            fix_hints: [],
            auto_normalized_applied: false,
          },
        },
      });
    } catch (err) {
      console.error('Failed to load demo case:', err);
    } finally {
      setCheckingQC(false);
    }
  };

  // Run live prediction
  const handleExecuteRun = async (confirmedCaution = false) => {
    if (!qcReport?.can_proceed) return;

    if (qcReport.requires_caution && !confirmedCaution) {
      setShowCautionModal(true);
      return;
    }

    setShowCautionModal(false);
    setRunningPrediction(true);

    try {
      const fd = new FormData();
      if (patientRef) fd.append('patient_ref', patientRef);
      if (files.rna.file) fd.append('rna', files.rna.file);
      if (files.mirna.file) fd.append('mirna', files.mirna.file);
      if (files.meth.file) fd.append('meth', files.meth.file);
      if (autoNormalize) fd.append('auto_normalize', 'true');
      if (confirmedCaution) fd.append('proceed_with_caution', 'true');

      const res = await api.predictLive(fd);
      const caseId = String(res.case_id || 'live_case');

      // Store in session storage for immediate Live Case screen access
      sessionStorage.setItem(`case_${caseId}`, JSON.stringify(res));

      // Navigate to live case screen
      navigate(`/live-case/${caseId}`);
    } catch (err: any) {
      console.error('Live prediction failed:', err);
      alert(`Prediction failed: ${err.message || 'Server error'}`);
    } finally {
      setRunningPrediction(false);
    }
  };

  const getStatusBadge = (status?: string) => {
    switch (status) {
      case 'PASSED':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
            <CheckCircle2 className="w-3.5 h-3.5" /> Pass
          </span>
        );
      case 'WARNING':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300">
            <AlertTriangle className="w-3.5 h-3.5" /> Warning
          </span>
        );
      case 'FAILED':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300">
            <XCircle className="w-3.5 h-3.5" /> Fail
          </span>
        );
      default:
        return null;
    }
  };

  return (
    <div className="pb-16 max-w-7xl mx-auto">
      <TopBar
        title="New Case Intake"
        subtitle="Upload patient assay data with automated quality control, scale verification, and live PAM50 prediction."
      />

      <div className="p-6 space-y-6">
        {/* Header Controls Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-slate-900 dark:text-white">
                Multi-Omics Intake & QC
              </h2>
              {isDemoData && (
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-100 text-amber-800 border border-amber-300">
                  DEMO DATA
                </span>
              )}
            </div>
            <p className="text-xs text-slate-500">
              Provide 1 to 3 modalities (CSV/TSV with <code>id,value</code>). Minimum 50% feature coverage required.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleLoadDemoCase}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-blue-700 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/50 dark:text-blue-300 dark:hover:bg-blue-900/60 rounded-xl transition-all border border-blue-200 dark:border-blue-800 shadow-sm"
            >
              <Sparkles className="w-4 h-4 text-blue-600 dark:text-blue-400" />
              Load example case
            </button>

            {/* Template Downloads */}
            <div className="flex items-center gap-1 border border-slate-200 dark:border-slate-700 rounded-xl p-1 bg-slate-50 dark:bg-slate-800 text-xs">
              <span className="text-[11px] text-slate-400 px-2 font-medium">Templates:</span>
              <a
                href={api.getTemplateUrl('rna')}
                download="rna_template.csv"
                className="px-2 py-1 rounded-lg hover:bg-white dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-mono text-[11px] flex items-center gap-1"
              >
                <Download className="w-3 h-3 text-slate-400" /> RNA
              </a>
              <a
                href={api.getTemplateUrl('mirna')}
                download="mirna_template.csv"
                className="px-2 py-1 rounded-lg hover:bg-white dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-mono text-[11px] flex items-center gap-1"
              >
                <Download className="w-3 h-3 text-slate-400" /> miRNA
              </a>
              <a
                href={api.getTemplateUrl('meth')}
                download="meth_template.csv"
                className="px-2 py-1 rounded-lg hover:bg-white dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-mono text-[11px] flex items-center gap-1"
              >
                <Download className="w-3 h-3 text-slate-400" /> Meth
              </a>
            </div>
          </div>
        </div>

        {/* Patient Reference & Explicit Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
          <div>
            <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
              Pseudonymous Patient Identifier (Ref):
            </label>
            <input
              type="text"
              placeholder="e.g. PATIENT-BRCA-042"
              value={patientRef}
              onChange={(e) => setPatientRef(e.target.value)}
              className="w-full text-xs font-mono px-3.5 py-2.5 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex flex-col justify-center space-y-1.5">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoNormalize}
                onChange={(e) => {
                  setAutoNormalize(e.target.checked);
                  if (files.rna.file || files.mirna.file) triggerQC();
                }}
                className="w-4 h-4 rounded text-blue-600 focus:ring-blue-500 border-slate-300"
              />
              <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
                Auto-normalize raw counts if detected (log2(TPM + 1))
              </span>
            </label>
            <p className="text-[11px] text-slate-500 pl-6">
              Never transform data silently: explicitly checks scale and records normalisation status in the diagnostic PDF report.
            </p>
          </div>
        </div>

        {/* Three Drag-and-Drop / QC Slots */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Slot 1: RNA-seq */}
          <SlotCard
            title="Bulk RNA-seq"
            modality="rna"
            expectedFeatures="5,000 canonical genes"
            fileState={files.rna}
            qcResult={qcReport?.results?.rna}
            checking={checkingQC}
            inputRef={rnaInputRef}
            onFileSelect={(f) => handleFileSelect('rna', f)}
          />

          {/* Slot 2: miRNA */}
          <SlotCard
            title="miRNA Sequencing"
            modality="mirna"
            expectedFeatures="1,881 mature microRNAs"
            fileState={files.mirna}
            qcResult={qcReport?.results?.mirna}
            checking={checkingQC}
            inputRef={mirnaInputRef}
            onFileSelect={(f) => handleFileSelect('mirna', f)}
          />

          {/* Slot 3: DNA Methylation */}
          <SlotCard
            title="DNA Methylation"
            modality="meth"
            expectedFeatures="3,957 high-variance probes"
            fileState={files.meth}
            qcResult={qcReport?.results?.meth}
            checking={checkingQC}
            inputRef={methInputRef}
            onFileSelect={(f) => handleFileSelect('meth', f)}
          />
        </div>

        {/* Summary Bar & Run Action */}
        <div className="bg-white dark:bg-slate-900 p-6 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-blue-600" />
              {qcReport
                ? qcReport.summary_message
                : 'Upload files above to begin live data-quality evaluation.'}
            </div>
            <div className="text-xs text-slate-500">
              {qcReport?.can_proceed
                ? 'All mandatory QC coverage checks passed. Case is ready for GraphGPS inductive inference.'
                : 'Run button will become enabled once at least 1 modality meets the 50% feature coverage threshold.'}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={!qcReport?.can_proceed || runningPrediction}
              onClick={() => handleExecuteRun(false)}
              className={`inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-bold shadow-md transition-all ${
                qcReport?.can_proceed && !runningPrediction
                  ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer shadow-blue-500/20'
                  : 'bg-slate-100 text-slate-400 dark:bg-slate-800 dark:text-slate-600 cursor-not-allowed shadow-none'
              }`}
            >
              {runningPrediction ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Running Inductive Model...
                </>
              ) : (
                <>
                  Run Live Case
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Caution Confirmation Modal */}
      <AnimatePresence>
        {showCautionModal && (
          <div className="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center p-4">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-white dark:bg-slate-900 rounded-2xl max-w-md w-full p-6 shadow-2xl border border-amber-200 dark:border-amber-900 space-y-4"
            >
              <div className="flex items-center gap-3 text-amber-600">
                <AlertTriangle className="w-6 h-6 shrink-0" />
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  Proceed With Caution?
                </h3>
              </div>

              <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                {qcReport?.summary_message || 'QC warnings were detected on one or more modalities.'}
              </p>

              <div className="p-3 bg-amber-50 dark:bg-amber-950/40 rounded-xl text-xs text-amber-800 dark:text-amber-300 border border-amber-200/50">
                Partial feature coverage or un-normalized scales may decrease model confidence and trigger diagnostic abstention.
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCautionModal(false)}
                  className="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300"
                >
                  Cancel & Review
                </button>
                <button
                  type="button"
                  onClick={() => handleExecuteRun(true)}
                  className="px-4 py-2 text-xs font-bold rounded-lg bg-amber-600 hover:bg-amber-700 text-white shadow-sm"
                >
                  Proceed with Caution
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

// =========================================================================
// QC SLOT CARD COMPONENT
// =========================================================================

interface SlotCardProps {
  title: string;
  modality: 'rna' | 'mirna' | 'meth';
  expectedFeatures: string;
  fileState: ModalityFileState;
  qcResult?: QCModalityItem;
  checking: boolean;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onFileSelect: (f: File) => void;
}

function SlotCard({
  title,
  modality,
  expectedFeatures,
  fileState,
  qcResult,
  checking,
  inputRef,
  onFileSelect,
}: SlotCardProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4 flex flex-col justify-between">
      <div className="space-y-3">
        {/* Slot Title & Expected Count */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white">{title}</h3>
            <span className="text-[11px] text-slate-400 font-mono">{expectedFeatures}</span>
          </div>
          {qcResult && (
            <div>
              {qcResult.status === 'PASSED' && (
                <span className="px-2 py-0.5 text-[11px] font-bold rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                  PASS
                </span>
              )}
              {qcResult.status === 'WARNING' && (
                <span className="px-2 py-0.5 text-[11px] font-bold rounded bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300">
                  WARN
                </span>
              )}
              {qcResult.status === 'FAILED' && (
                <span className="px-2 py-0.5 text-[11px] font-bold rounded bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300">
                  FAIL
                </span>
              )}
            </div>
          )}
        </div>

        {/* Drop Zone or QC Result View */}
        {!fileState.file ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[160px] ${
              isDragOver
                ? 'border-blue-500 bg-blue-50/50 dark:bg-blue-950/20'
                : 'border-slate-200 dark:border-slate-700 hover:border-blue-400 dark:hover:border-blue-600 bg-slate-50/50 dark:bg-slate-800/40'
            }`}
          >
            <input
              type="file"
              ref={inputRef}
              accept=".csv,.tsv,.txt"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  onFileSelect(e.target.files[0]);
                }
              }}
            />
            <UploadCloud className="w-8 h-8 text-blue-500 mb-2" />
            <div className="text-xs font-semibold text-slate-700 dark:text-slate-200">
              Drag & drop CSV/TSV
            </div>
            <div className="text-[11px] text-slate-400 mt-0.5">or click to browse</div>
          </div>
        ) : (
          <div className="space-y-3">
            {/* File Info & Upload Progress */}
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <div className="font-semibold text-slate-800 dark:text-slate-200 truncate max-w-[180px] flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5 text-blue-600 shrink-0" />
                  {fileState.name}
                </div>
                <span className="font-mono text-slate-400 text-[11px]">{fileState.size}</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-700 h-1.5 rounded-full overflow-hidden">
                <div className="bg-blue-600 h-full rounded-full transition-all duration-300 w-full" />
              </div>
            </div>

            {/* QC Metrics Card */}
            {qcResult && (
              <div className="p-3.5 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Feature Coverage:</span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white">
                    {(qcResult.coverage_pct * 100).toFixed(1)}% ({qcResult.mapped_count}/{qcResult.total_expected})
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Missing Features:</span>
                  <span className="font-mono text-slate-700 dark:text-slate-300">
                    {(qcResult.missing_pct * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Detected Scale:</span>
                  <span className="font-mono text-xs px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-slate-200">
                    {qcResult.detected_scale}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-500">Duplicate IDs:</span>
                  <span className={`font-mono ${qcResult.duplicate_ids_count > 0 ? 'text-amber-600 font-bold' : 'text-slate-600'}`}>
                    {qcResult.duplicate_ids_count}
                  </span>
                </div>

                {/* Fix hints */}
                {qcResult.fix_hints.length > 0 && (
                  <div className="pt-2 border-t border-slate-100 dark:border-slate-700 text-[11px] text-amber-700 dark:text-amber-400 space-y-1">
                    <span className="font-semibold flex items-center gap-1">
                      <Info className="w-3 h-3" /> Fix Hint:
                    </span>
                    {qcResult.fix_hints.map((hint, idx) => (
                      <div key={idx} className="leading-tight pl-4">
                        &bull; {hint}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {fileState.file && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="w-full py-1.5 text-[11px] text-slate-500 hover:text-blue-600 text-center font-medium border-t border-slate-100 dark:border-slate-800 mt-2"
        >
          Replace file
        </button>
      )}
    </div>
  );
}
