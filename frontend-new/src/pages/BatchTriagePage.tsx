import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard } from '@/components/ui';
import { api } from '@/api/client';
import type { BatchResponse, BatchCardItem, BatchSummary } from '@/lib/types';
import {
  UploadCloud,
  FileSpreadsheet,
  Download,
  Sparkles,
  Filter,
  CheckCircle2,
  AlertTriangle,
  Clock,
  DollarSign,
  Layers,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  Search,
  ExternalLink,
  ChevronRight,
  Calculator,
  XCircle,
} from 'lucide-react';

const SUBTYPE_CHIPS: Record<string, { bg: string; text: string; border: string }> = {
  LumA: {
    bg: 'bg-emerald-500/10 dark:bg-emerald-500/20',
    text: 'text-emerald-700 dark:text-emerald-300',
    border: 'border-emerald-500/30',
  },
  LumB: {
    bg: 'bg-blue-500/10 dark:bg-blue-500/20',
    text: 'text-blue-700 dark:text-blue-300',
    border: 'border-blue-500/30',
  },
  Her2: {
    bg: 'bg-purple-500/10 dark:bg-purple-500/20',
    text: 'text-purple-700 dark:text-purple-300',
    border: 'border-purple-500/30',
  },
  Basal: {
    bg: 'bg-rose-500/10 dark:bg-rose-500/20',
    text: 'text-rose-700 dark:text-rose-300',
    border: 'border-rose-500/30',
  },
  Normal: {
    bg: 'bg-amber-500/10 dark:bg-amber-500/20',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-500/30',
  },
};

export function BatchTriagePage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [batchData, setBatchData] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [processingProgress, setProcessingProgress] = useState<number | null>(null);
  const [processedCount, setProcessedCount] = useState<number>(0);
  const [totalBatchSize, setTotalBatchSize] = useState<number>(0);
  const [isDemo, setIsDemo] = useState(false);

  // User input for hands-on-time saved (NO default invented value!)
  const [minutesPerCase, setMinutesPerCase] = useState<string>('');

  // Filters
  const [selectedSubtype, setSelectedSubtype] = useState<string>('all');
  const [selectedTier, setSelectedTier] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Handle Manifest / Zip Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setIsDemo(false);
    setProcessingProgress(20);
    setProcessedCount(0);
    setTotalBatchSize(0);

    try {
      const fd = new FormData();
      if (file.name.endsWith('.zip')) {
        fd.append('zip_file', file);
      } else {
        fd.append('manifest', file);
      }

      setProcessingProgress(45);
      const res = await api.createBatch(fd, false);
      setTotalBatchSize(res.total_cases);
      setProcessedCount(res.total_cases);
      setProcessingProgress(100);
      setBatchData(res);
    } catch (err: any) {
      console.error('Batch upload failed:', err);
      alert(`Batch processing failed: ${err.message || 'Server error'}`);
    } finally {
      setLoading(false);
      setTimeout(() => setProcessingProgress(null), 800);
    }
  };

  // Run Demo Batch (Real out-of-fold cohort patients badged demo data)
  const handleRunDemoBatch = async () => {
    setLoading(true);
    setIsDemo(true);
    setProcessingProgress(15);
    setProcessedCount(0);
    setTotalBatchSize(25);

    try {
      // Simulate live counts incrementing for visual polish
      const timer = setInterval(() => {
        setProcessedCount((prev) => {
          if (prev >= 25) {
            clearInterval(timer);
            return 25;
          }
          return prev + 5;
        });
        setProcessingProgress((prev) => (prev ? Math.min(prev + 15, 90) : 30));
      }, 150);

      const res = await api.createBatch(undefined, true);
      clearInterval(timer);
      setTotalBatchSize(res.total_cases);
      setProcessedCount(res.total_cases);
      setProcessingProgress(100);
      setBatchData(res);
    } catch (err: any) {
      console.error('Demo batch failed:', err);
      alert(`Demo batch failed: ${err.message || 'Server error'}`);
    } finally {
      setLoading(false);
      setTimeout(() => setProcessingProgress(null), 800);
    }
  };

  // Export CSV of the board
  const handleExportCsv = () => {
    if (!batchData?.batch_id) return;
    const url = api.getBatchResultsUrl(batchData.batch_id);
    const link = document.createElement('a');
    link.href = url;
    link.download = `BioNexus_Batch_${batchData.batch_id}_Triage.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Calculate hands-on time saved ONLY if user has entered minutes-per-case
  const parsedMinutes = parseFloat(minutesPerCase);
  const hasUserEnteredTime = !isNaN(parsedMinutes) && parsedMinutes > 0;
  let handsOnHoursSaved: number | null = null;
  if (hasUserEnteredTime && batchData?.summary) {
    const autoCount = batchData.summary.auto_reported_count;
    handsOnHoursSaved = parseFloat(((autoCount * parsedMinutes) / 60).toFixed(1));
  }

  // Filter cards by subtype and tier
  const filterCards = (cards?: BatchCardItem[]) => {
    if (!cards) return [];
    return cards.filter((card) => {
      if (selectedSubtype !== 'all' && card.predicted_subtype !== selectedSubtype) {
        return false;
      }
      if (selectedTier !== 'all' && card.confidence_tier.toLowerCase() !== selectedTier.toLowerCase()) {
        return false;
      }
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          card.patient_ref.toLowerCase().includes(q) ||
          card.case_id.toLowerCase().includes(q) ||
          card.predicted_subtype.toLowerCase().includes(q)
        );
      }
      return true;
    });
  };

  const autoReportCards = filterCards(batchData?.lanes?.auto_report);
  const clinicianReviewCards = filterCards(batchData?.lanes?.clinician_review);
  const needsMoreDataCards = filterCards(batchData?.lanes?.needs_more_data);

  return (
    <div className="pb-16 max-w-7xl mx-auto space-y-6">
      <TopBar
        title="Batch Triage Board"
        subtitle="High-throughput intake, automated confidence gating, and clinical review routing"
      />

      {/* Control Banner: Upload Manifest / Zip or Run Demo Batch */}
      <div className="p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              Multi-Case Cohort Intake
            </h2>
            {isDemo && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide uppercase bg-blue-100 text-blue-700 dark:bg-blue-900/60 dark:text-blue-300">
                Demo Cohort Active
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 max-w-xl">
            Upload a batch manifest CSV (patient_ref, rna_file, ...) or zip archive to triage cases into automated reporting vs manual pathologist review.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.tsv,.zip"
            onChange={handleFileUpload}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors disabled:opacity-50"
          >
            <UploadCloud className="w-4 h-4 text-blue-600" />
            <span>Upload Manifest / Zip</span>
          </button>

          <button
            onClick={handleRunDemoBatch}
            disabled={loading}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-500/20 hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4 text-amber-300" />
            <span>Run Demo Batch (Out-of-Fold Cohort)</span>
          </button>
        </div>
      </div>

      {/* Progress Bar with Live Counts */}
      {processingProgress !== null && (
        <div className="p-4 rounded-xl bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900 space-y-2">
          <div className="flex justify-between text-xs font-semibold text-blue-900 dark:text-blue-200">
            <span className="flex items-center gap-2">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Processing batch: {processedCount} of {totalBatchSize || 25} cases evaluated...
            </span>
            <span className="font-mono">{processingProgress}%</span>
          </div>
          <div className="w-full bg-blue-200 dark:bg-blue-900/60 h-2 rounded-full overflow-hidden">
            <div
              className="bg-blue-600 h-full rounded-full transition-all duration-300"
              style={{ width: `${processingProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Summary Header Metrics */}
      {batchData?.summary && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {/* Total Cases */}
          <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">
              Total Cases
            </span>
            <div className="text-2xl font-black font-mono text-slate-900 dark:text-white mt-1">
              {batchData.summary.total_cases}
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">Ingested batch volume</p>
          </div>

          {/* % Auto-Reported */}
          <div className="p-4 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800">
            <span className="text-[11px] font-medium text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
              Auto-Reported
            </span>
            <div className="text-2xl font-black font-mono text-emerald-800 dark:text-emerald-300 mt-1">
              {(batchData.summary.auto_reported_pct * 100).toFixed(0)}%
            </div>
            <p className="text-[10px] text-emerald-600/80 mt-0.5">
              {batchData.summary.auto_reported_count} cases (Conf ≥ 80%)
            </p>
          </div>

          {/* % Clinician Review */}
          <div className="p-4 rounded-xl bg-amber-50/50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800">
            <span className="text-[11px] font-medium text-amber-700 dark:text-amber-400 uppercase tracking-wider">
              Clinician Review
            </span>
            <div className="text-2xl font-black font-mono text-amber-800 dark:text-amber-300 mt-1">
              {(batchData.summary.clinician_review_pct * 100).toFixed(0)}%
            </div>
            <p className="text-[10px] text-amber-600/80 mt-0.5">
              {batchData.summary.clinician_review_count} cases (60%–79%)
            </p>
          </div>

          {/* Average Cost Per Case */}
          <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
            <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">
              Avg Cost / Case
            </span>
            <div className="text-2xl font-black font-mono text-slate-900 dark:text-white mt-1">
              ${batchData.summary.average_cost_per_case.toFixed(0)}
            </div>
            <p className="text-[10px] text-slate-400 mt-0.5">Assay panel composite</p>
          </div>

          {/* Hands-On Time Saved (Only shown if user enters minutes-per-case) */}
          <div className="p-4 rounded-xl bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-blue-700 dark:text-blue-400 uppercase tracking-wider">
                  Hands-On Time Saved
                </span>
                <Clock className="w-3.5 h-3.5 text-blue-500" />
              </div>

              {hasUserEnteredTime && handsOnHoursSaved !== null ? (
                <div className="mt-1">
                  <div className="text-2xl font-black font-mono text-blue-800 dark:text-blue-300">
                    {handsOnHoursSaved} hrs
                  </div>
                  <p className="text-[10px] text-blue-600/80">
                    At {minutesPerCase} min/case saved on {batchData.summary.auto_reported_count} cases
                  </p>
                </div>
              ) : (
                <div className="mt-1">
                  <span className="text-xs text-slate-500 italic block">Enter min/case below</span>
                  <p className="text-[10px] text-slate-400 mt-0.5">No invented default</p>
                </div>
              )}
            </div>

            {/* Input to provide minutes per case */}
            <div className="mt-2 pt-2 border-t border-blue-200/60 dark:border-blue-800/60 flex items-center gap-1.5">
              <input
                type="number"
                min="1"
                max="300"
                placeholder="Mins/case..."
                value={minutesPerCase}
                onChange={(e) => setMinutesPerCase(e.target.value)}
                className="w-full px-2 py-1 text-xs rounded bg-white dark:bg-slate-900 border border-blue-200 dark:border-blue-800 font-mono text-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <span className="text-[10px] text-blue-600 font-semibold">min</span>
            </div>
          </div>
        </div>
      )}

      {/* Filter and Export Toolbar */}
      {batchData && (
        <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800">
          <div className="flex flex-wrap items-center gap-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-400" />
              <input
                type="text"
                placeholder="Filter patient ref..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 rounded-lg text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Subtype Filter */}
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <Filter className="w-3.5 h-3.5" />
              <span>Subtype:</span>
              <select
                value={selectedSubtype}
                onChange={(e) => setSelectedSubtype(e.target.value)}
                className="px-2 py-1 rounded-lg text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
              >
                <option value="all">All Subtypes</option>
                <option value="LumA">Luminal A</option>
                <option value="LumB">Luminal B</option>
                <option value="Her2">HER2-enriched</option>
                <option value="Basal">Basal-like</option>
                <option value="Normal">Normal-like</option>
              </select>
            </div>

            {/* Tier Filter */}
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <span>Tier:</span>
              <select
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                className="px-2 py-1 rounded-lg text-xs bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-900 dark:text-white"
              >
                <option value="all">All Tiers</option>
                <option value="high">High (≥80%)</option>
                <option value="medium">Medium (60-79%)</option>
                <option value="low">Low (&lt;60%)</option>
              </select>
            </div>
          </div>

          {/* Export CSV Button */}
          <button
            onClick={handleExportCsv}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Board CSV</span>
          </button>
        </div>
      )}

      {/* 3-Lane Kanban Board */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Lane 1: Auto-report (High confidence >= 80%) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              <h3 className="text-xs font-bold text-emerald-900 dark:text-emerald-200 uppercase tracking-wider">
                Auto-Report
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-mono font-bold bg-emerald-200 dark:bg-emerald-900 text-emerald-800 dark:text-emerald-200">
              {autoReportCards.length}
            </span>
          </div>

          <div className="space-y-3 min-h-[400px]">
            {autoReportCards.map((card) => (
              <TriageCard key={card.case_id} card={card} onOpen={() => navigate(`/live-case/${card.case_id}`)} />
            ))}
            {autoReportCards.length === 0 && (
              <div className="p-8 text-center text-slate-400 bg-white dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-800 text-xs">
                {batchData ? 'No matching cases in this lane' : 'Load batch to view cases'}
              </div>
            )}
          </div>
        </div>

        {/* Lane 2: Clinician Review (Medium confidence 60-79%) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-800">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <h3 className="text-xs font-bold text-amber-900 dark:text-amber-200 uppercase tracking-wider">
                Clinician Review
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-mono font-bold bg-amber-200 dark:bg-amber-900 text-amber-800 dark:text-amber-200">
              {clinicianReviewCards.length}
            </span>
          </div>

          <div className="space-y-3 min-h-[400px]">
            {clinicianReviewCards.map((card) => (
              <TriageCard key={card.case_id} card={card} onOpen={() => navigate(`/live-case/${card.case_id}`)} />
            ))}
            {clinicianReviewCards.length === 0 && (
              <div className="p-8 text-center text-slate-400 bg-white dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-800 text-xs">
                {batchData ? 'No matching cases in this lane' : 'Load batch to view cases'}
              </div>
            )}
          </div>
        </div>

        {/* Lane 3: Needs More Data (Low confidence < 60% or missing modalities) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800">
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-rose-600 dark:text-rose-400" />
              <h3 className="text-xs font-bold text-rose-900 dark:text-rose-200 uppercase tracking-wider">
                Needs More Data
              </h3>
            </div>
            <span className="px-2 py-0.5 rounded-full text-xs font-mono font-bold bg-rose-200 dark:bg-rose-900 text-rose-800 dark:text-rose-200">
              {needsMoreDataCards.length}
            </span>
          </div>

          <div className="space-y-3 min-h-[400px]">
            {needsMoreDataCards.map((card) => (
              <TriageCard key={card.case_id} card={card} onOpen={() => navigate(`/live-case/${card.case_id}`)} />
            ))}
            {needsMoreDataCards.length === 0 && (
              <div className="p-8 text-center text-slate-400 bg-white dark:bg-slate-900/40 rounded-xl border border-dashed border-slate-200 dark:border-slate-800 text-xs">
                {batchData ? 'No matching cases in this lane' : 'Load batch to view cases'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Kanban Card Item
function TriageCard({ card, onOpen }: { card: BatchCardItem; onOpen: () => void }) {
  const chipStyle = SUBTYPE_CHIPS[card.predicted_subtype] || SUBTYPE_CHIPS.LumA;
  const confPct = Math.round(card.confidence * 100);

  return (
    <motion.div
      whileHover={{ y: -2 }}
      onClick={onOpen}
      className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md hover:border-blue-400 dark:hover:border-blue-600 transition-all cursor-pointer space-y-3 group"
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono font-bold text-sm text-slate-900 dark:text-white group-hover:text-blue-600 transition-colors">
              {card.patient_ref}
            </span>
            {card.demo_data && (
              <span className="px-1.5 py-0.2 rounded text-[9px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-500">
                demo
              </span>
            )}
          </div>
          <span className="text-[10px] text-slate-400 font-mono">{card.case_id}</span>
        </div>

        <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 transition-colors" />
      </div>

      <div className="flex items-center justify-between">
        <span
          className={`px-2.5 py-0.5 rounded-md text-xs font-bold border ${chipStyle.bg} ${chipStyle.text} ${chipStyle.border}`}
        >
          {card.predicted_subtype}
        </span>
        <div className="text-right">
          <span className="font-mono font-bold text-sm text-slate-900 dark:text-white">
            {confPct}%
          </span>
          <span className="text-[10px] text-slate-400 block">confidence</span>
        </div>
      </div>

      {/* Data types used chips */}
      <div className="pt-2 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between">
        <div className="flex flex-wrap gap-1">
          {card.data_types_used.map((dt) => (
            <span
              key={dt}
              className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300"
            >
              {dt.replace('-seq', '')}
            </span>
          ))}
        </div>
        <span className="text-[10px] font-mono text-slate-400">${card.estimated_cost}</span>
      </div>
    </motion.div>
  );
}
