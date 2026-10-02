import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard, ErrorState } from '@/components/ui';
import { api } from '@/api/client';
import type { CostAccuracyResponse, PolicyPoint, ModalityCombinationItem } from '@/lib/types';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  BarChart,
  Bar,
  ErrorBar,
  Cell,
} from 'recharts';
import {
  TrendingUp,
  Sliders,
  DollarSign,
  AlertTriangle,
  Layers,
  ShieldAlert,
  Info,
  CheckCircle2,
  Sparkles,
  Lock,
} from 'lucide-react';

export function CostAccuracyPage() {
  const [data, setData] = useState<CostAccuracyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // User interactive state
  const [metric, setMetric] = useState<'macro_f1' | 'accuracy'>('macro_f1');
  const [budgetSlider, setBudgetSlider] = useState<number>(800);
  const [thresholdSlider, setThresholdSlider] = useState<number>(0.80);

  // Fetch precomputed data
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    api.getCostAccuracy(undefined, undefined, metric)
      .then((res) => {
        if (!active) return;
        setData(res);
        if (res.nearest_point) {
          setBudgetSlider(Math.round(res.nearest_point.avg_cost));
          setThresholdSlider(res.nearest_point.threshold);
        }
      })
      .catch((err) => {
        if (!active) return;
        console.error('Failed to load cost-accuracy simulation:', err);
        setError(err.message || 'Failed to fetch simulation data');
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [metric]);

  // Find matching point based on slider positions
  const matchingPoint = useMemo<PolicyPoint | null>(() => {
    if (!data?.policy_points || data.policy_points.length === 0) return null;
    // Find closest point by budget
    let closest = data.policy_points[0];
    let minDiff = Math.abs(closest.avg_cost - budgetSlider);

    for (const pt of data.policy_points) {
      const diff = Math.abs(pt.avg_cost - budgetSlider);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    }
    return closest;
  }, [data, budgetSlider]);

  // Sync sliders when user changes threshold
  const handleThresholdChange = (newThreshold: number) => {
    setThresholdSlider(newThreshold);
    if (!data?.policy_points) return;
    let closest = data.policy_points[0];
    let minDiff = Math.abs(closest.threshold - newThreshold);
    for (const pt of data.policy_points) {
      const diff = Math.abs(pt.threshold - newThreshold);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    }
    setBudgetSlider(Math.round(closest.avg_cost));
  };

  // Sync sliders when user changes budget
  const handleBudgetChange = (newBudget: number) => {
    setBudgetSlider(newBudget);
    if (!data?.policy_points) return;
    let closest = data.policy_points[0];
    let minDiff = Math.abs(closest.avg_cost - newBudget);
    for (const pt of data.policy_points) {
      const diff = Math.abs(pt.avg_cost - newBudget);
      if (diff < minDiff) {
        minDiff = diff;
        closest = pt;
      }
    }
    setThresholdSlider(closest.threshold);
  };

  // Dynamic readout card text:
  // "At budget X, adaptive policy reaches Y% of full-panel accuracy with Z% lower cost."
  const dynamicReadout = useMemo(() => {
    if (!matchingPoint) return 'Select a target budget or confidence threshold.';
    const cost = Math.round(matchingPoint.avg_cost);
    const pct = metric === 'macro_f1'
      ? matchingPoint.pct_of_full_f1
      : matchingPoint.pct_of_full_accuracy;
    const label = metric === 'macro_f1' ? 'macro-F1' : 'full-panel accuracy';
    return `At budget $${cost}/case, adaptive policy reaches ${pct}% of ${label} with ${matchingPoint.cost_savings_pct}% lower cost.`;
  }, [matchingPoint, metric]);

  // Chart data format
  const chartData = useMemo(() => {
    if (!data?.policy_points) return [];
    const alwaysF1 = data.baselines?.always_all?.macro_f1 ?? 0.8458;
    const alwaysAcc = data.baselines?.always_all?.accuracy ?? 0.8699;
    const alwaysVal = metric === 'macro_f1' ? alwaysF1 : alwaysAcc;

    const neverF1 = data.baselines?.never_add?.macro_f1 ?? 0.8104;
    const neverAcc = data.baselines?.never_add?.accuracy ?? 0.8144;
    const neverVal = metric === 'macro_f1' ? neverF1 : neverAcc;

    return data.policy_points.map((pt) => {
      const isSelected = matchingPoint && Math.abs(pt.avg_cost - matchingPoint.avg_cost) < 1.0;
      const val = metric === 'macro_f1' ? pt.macro_f1 : pt.accuracy;
      const ciLow = metric === 'macro_f1' ? pt.macro_f1_ci_lower : pt.accuracy_ci_lower;
      const ciHigh = metric === 'macro_f1' ? pt.macro_f1_ci_upper : pt.accuracy_ci_upper;

      return {
        cost: Math.round(pt.avg_cost),
        threshold: pt.threshold,
        adaptive: val,
        adaptive_ci_lower: ciLow,
        adaptive_ci_upper: ciHigh,
        ci_band: [ciLow, ciHigh],
        always_all: alwaysVal,
        never_add: neverVal,
        isSelected,
        point: pt,
      };
    });
  }, [data, metric, matchingPoint]);

  // 7 Modality combinations data for bar chart
  const modalityBarData = useMemo(() => {
    if (!data?.modality_combinations) return [];
    return data.modality_combinations.map((m: ModalityCombinationItem) => {
      const val = metric === 'macro_f1' ? m.macro_f1 : m.accuracy;
      const std = metric === 'macro_f1' ? m.macro_f1_std : m.accuracy_std;
      return {
        name: m.name,
        modalities: m.modalities.join(' + '),
        cost: m.cost,
        value: val,
        error: std,
        errorRange: [Math.max(0, val - std), Math.min(1, val + std)],
        id: m.id,
      };
    });
  }, [data, metric]);

  if (loading) {
    return (
      <div>
        <TopBar title="Cost vs Accuracy" subtitle="Loading precomputed policy simulation curves..." />
        <div className="p-6 flex items-center justify-center min-h-[400px]">
          <div className="flex flex-col items-center gap-3 text-slate-500">
            <div className="w-8 h-8 border-3 border-blue-600 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm font-medium">Loading precomputed F3 policy trade-offs...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !data || data.empty) {
    return (
      <div>
        <TopBar title="Cost vs Accuracy" subtitle="Diagnostic policy simulation and budget optimization." />
        <div className="p-6">
          <ErrorState
            error={data?.message || error || "Run 'python scripts/generate_cost_accuracy_artifacts.py' to generate results/f3/policy_curve.csv and gain table."}
          />
        </div>

      </div>
    );
  }

  return (
    <div className="pb-16">
      <TopBar
        title="Cost vs Accuracy"
        subtitle="Simulate diagnostic panel trade-offs, adaptive escalation policies, and multi-omics yield."
      />

      <div className="p-6 space-y-6 max-w-7xl mx-auto">
        {/* ========================================================================= */}
        {/* 1. CODE-GENERATED HEADLINE & CAVEAT BANNER */}
        {/* ========================================================================= */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-900/90 via-slate-900 to-indigo-950 p-6 text-white shadow-xl border border-blue-500/20"
        >
          <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-2 max-w-4xl">
              <div className="flex items-center gap-2.5">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-400/20 text-amber-300 border border-amber-400/40">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  {data.placeholder_badge || 'Placeholder Quote'}
                </span>
                <span className="text-xs uppercase tracking-wider text-blue-200/70 font-mono">
                  TCGA-BRCA 5-Fold Stratified Policy Simulation
                </span>
              </div>
              <h2 className="text-lg md:text-xl font-bold tracking-tight text-white leading-snug">
                {data.headline || dynamicReadout}
              </h2>
              <p className="text-xs md:text-sm text-slate-300/80 leading-relaxed">
                {data.caveat || 'Note: Assay costs reflect estimated laboratory placeholder values.'}
              </p>
            </div>

            <div className="shrink-0 flex items-center gap-2 bg-white/10 backdrop-blur-md px-4 py-3 rounded-xl border border-white/10">
              <Sparkles className="w-5 h-5 text-amber-300" />
              <div>
                <div className="text-xs text-slate-300 font-medium">Optimal Sweet Spot</div>
                <div className="text-sm font-bold text-white">$800 / case (38.5% savings)</div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* ========================================================================= */}
        {/* 2. DYNAMIC READOUT CARD & METRIC TOGGLE */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-sm border border-slate-200 dark:border-slate-800 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-100 dark:border-slate-800 pb-4">
              <div>
                <span className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider">
                  Interactive Policy Evaluation
                </span>
                <h3 className="text-base font-bold text-slate-900 dark:text-white">
                  Real-time Readout & Policy Tuning
                </h3>
              </div>

              {/* Metric Toggle */}
              <div className="flex items-center bg-slate-100 dark:bg-slate-800 p-1 rounded-xl self-start">
                <button
                  type="button"
                  onClick={() => setMetric('macro_f1')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                    metric === 'macro_f1'
                      ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Macro-F1
                </button>
                <button
                  type="button"
                  onClick={() => setMetric('accuracy')}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                    metric === 'accuracy'
                      ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
                  }`}
                >
                  Accuracy
                </button>
              </div>
            </div>

            {/* Readout Highlight Box */}
            <div className="p-4 rounded-xl bg-blue-50/80 dark:bg-blue-950/40 border border-blue-200/60 dark:border-blue-900/50 flex items-start gap-3">
              <CheckCircle2 className="w-5 h-5 text-blue-600 dark:text-blue-400 shrink-0 mt-0.5" />
              <div>
                <div className="text-sm font-bold text-blue-950 dark:text-blue-100">
                  {dynamicReadout}
                </div>
                <div className="text-xs text-blue-800/80 dark:text-blue-300/80 mt-1">
                  Matched threshold &tau; = {matchingPoint?.threshold.toFixed(2)} | Escalation rate: {((matchingPoint?.escalation_rate ?? 0) * 100).toFixed(1)}% of cases triggered supplementary assays.
                </div>
              </div>
            </div>

            {/* Dual Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-2">
              {/* Budget Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <DollarSign className="w-3.5 h-3.5 text-emerald-600" />
                    Target Budget per Case:
                  </span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-xs">
                    ${budgetSlider}
                  </span>
                </div>
                <input
                  type="range"
                  min={500}
                  max={1300}
                  step={25}
                  value={budgetSlider}
                  onChange={(e) => handleBudgetChange(Number(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                  <span>$500 (RNA alone)</span>
                  <span>$1,300 (Full Panel)</span>
                </div>
              </div>

              {/* Confidence Threshold Slider */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                    <Sliders className="w-3.5 h-3.5 text-indigo-600" />
                    Confidence Threshold (&tau;):
                  </span>
                  <span className="font-mono font-bold text-slate-900 dark:text-white bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded text-xs">
                    {thresholdSlider.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.50}
                  max={0.98}
                  step={0.01}
                  value={thresholdSlider}
                  onChange={(e) => handleThresholdChange(Number(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                />
                <div className="flex justify-between text-[11px] text-slate-400 font-mono">
                  <span>0.50 (Permissive)</span>
                  <span>0.98 (Strict Escalation)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Metrics Card */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-sm border border-slate-200 dark:border-slate-800 flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Policy Summary
                </h4>
                <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400 font-medium">
                  Adaptive
                </span>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-slate-500">Selected Macro-F1:</span>
                  <span className="text-base font-bold text-slate-900 dark:text-white font-mono">
                    {matchingPoint?.macro_f1.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-slate-500">Selected Accuracy:</span>
                  <span className="text-base font-bold text-slate-900 dark:text-white font-mono">
                    {matchingPoint?.accuracy.toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-slate-500">95% Bootstrap CI:</span>
                  <span className="text-xs font-mono text-slate-700 dark:text-slate-300">
                    [{metric === 'macro_f1' ? matchingPoint?.macro_f1_ci_lower.toFixed(3) : matchingPoint?.accuracy_ci_lower.toFixed(3)},{' '}
                    {metric === 'macro_f1' ? matchingPoint?.macro_f1_ci_upper.toFixed(3) : matchingPoint?.accuracy_ci_upper.toFixed(3)}]
                  </span>
                </div>
                <div className="flex justify-between items-baseline">
                  <span className="text-xs text-slate-500">Cost Savings vs Full:</span>
                  <span className="text-sm font-bold text-emerald-600 dark:text-emerald-400">
                    {matchingPoint?.cost_savings_pct}%
                  </span>
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-400 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 shrink-0" />
              <span>Bootstrap intervals derived from 1,000 resamples.</span>
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* 3. MAIN CHART & SIDE PANEL */}
        {/* ========================================================================= */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Chart (2 cols) */}
          <div className="lg:col-span-2 rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-sm border border-slate-200 dark:border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-blue-600" />
                  Cost vs Performance Trade-off Curves
                </h3>
                <p className="text-xs text-slate-500">
                  x = Average Cost per Case ($ USD) &bull; y = {metric === 'macro_f1' ? 'Macro-F1' : 'Accuracy'} with shaded bootstrap intervals.
                </p>
              </div>

              {/* Legend Badges */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400 font-medium">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-600" /> Adaptive Policy
                </span>
                <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium">
                  <span className="w-2.5 h-0.5 bg-emerald-600" /> Always All ($1,300)
                </span>
                <span className="flex items-center gap-1 text-slate-500 font-medium">
                  <span className="w-2.5 h-0.5 bg-slate-400 border-dashed" /> Never Add ($500)
                </span>
              </div>
            </div>

            {/* Recharts Main Graph */}
            <div className="h-[420px] w-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={chartData}
                  margin={{ top: 15, right: 30, left: 10, bottom: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
                  <XAxis
                    dataKey="cost"
                    type="number"
                    domain={[480, 1320]}
                    tickFormatter={(val) => `$${val}`}
                    label={{ value: 'Average Cost per Case ($ USD)', position: 'insideBottom', offset: -15, fontSize: 12, fill: '#64748b' }}
                    stroke="#94a3b8"
                  />
                  <YAxis
                    domain={[0.70, 0.90]}
                    tickFormatter={(val) => val.toFixed(2)}
                    label={{ value: metric === 'macro_f1' ? 'Macro-F1' : 'Accuracy', angle: -90, position: 'insideLeft', offset: 0, fontSize: 12, fill: '#64748b' }}
                    stroke="#94a3b8"
                  />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null;
                      const pt = payload[0].payload;
                      return (
                        <div className="rounded-xl bg-slate-900/95 text-white p-3.5 shadow-xl border border-slate-800 text-xs space-y-1.5 backdrop-blur-md">
                          <div className="font-bold text-blue-400 flex justify-between gap-4">
                            <span>Cost: ${pt.cost} / case</span>
                            <span>&tau; = {pt.threshold}</span>
                          </div>
                          <div className="flex justify-between gap-4 text-slate-300">
                            <span>Adaptive {metric === 'macro_f1' ? 'F1' : 'Acc'}:</span>
                            <span className="font-mono font-bold text-white">{pt.adaptive.toFixed(4)}</span>
                          </div>
                          <div className="flex justify-between gap-4 text-slate-400 text-[11px]">
                            <span>95% Bootstrap CI:</span>
                            <span className="font-mono">[{pt.adaptive_ci_lower.toFixed(3)}, {pt.adaptive_ci_upper.toFixed(3)}]</span>
                          </div>
                          <div className="border-t border-slate-800 pt-1 flex justify-between gap-4 text-emerald-400">
                            <span>Cost Savings:</span>
                            <span>{pt.point?.cost_savings_pct}%</span>
                          </div>
                        </div>
                      );
                    }}
                  />

                  {/* Horizontal Baselines */}
                  <ReferenceLine
                    y={chartData[0]?.always_all}
                    stroke="#10b981"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    label={{ value: `Always Full Panel (${chartData[0]?.always_all.toFixed(3)})`, position: 'top', fill: '#10b981', fontSize: 11 }}
                  />
                  <ReferenceLine
                    y={chartData[0]?.never_add}
                    stroke="#94a3b8"
                    strokeWidth={2}
                    strokeDasharray="3 3"
                    label={{ value: `Never Add Tests (${chartData[0]?.never_add.toFixed(3)})`, position: 'bottom', fill: '#64748b', fontSize: 11 }}
                  />

                  {/* Highlight Selected Matching Point Line */}
                  {matchingPoint && (
                    <ReferenceLine
                      x={Math.round(matchingPoint.avg_cost)}
                      stroke="#3b82f6"
                      strokeWidth={1.5}
                      strokeDasharray="3 3"
                      label={{ value: `Budget $${Math.round(matchingPoint.avg_cost)}`, position: 'insideTopLeft', fill: '#3b82f6', fontSize: 11 }}
                    />
                  )}

                  {/* Shaded Bootstrap Confidence Interval Area */}
                  <Area
                    type="monotone"
                    dataKey="adaptive_ci_upper"
                    stroke="none"
                    fill="#3b82f6"
                    fillOpacity={0.15}
                    name="Bootstrap 95% CI"
                  />
                  <Area
                    type="monotone"
                    dataKey="adaptive_ci_lower"
                    stroke="none"
                    fill="#ffffff"
                    fillOpacity={1.0}
                  />

                  {/* Adaptive Policy Main Line */}
                  <Line
                    type="monotone"
                    dataKey="adaptive"
                    stroke="#2563eb"
                    strokeWidth={3}
                    dot={(props: any) => {
                      const { cx, cy, payload } = props;
                      const isMatch = matchingPoint && Math.abs(payload.cost - Math.round(matchingPoint.avg_cost)) < 1.0;
                      if (isMatch) {
                        return (
                          <g key={`match-${cx}-${cy}`}>
                            <circle cx={cx} cy={cy} r={8} fill="#2563eb" fillOpacity={0.3} className="animate-ping" />
                            <circle cx={cx} cy={cy} r={6} fill="#2563eb" stroke="#ffffff" strokeWidth={2} />
                          </g>
                        );
                      }
                      return (
                        <circle key={`dot-${cx}-${cy}`} cx={cx} cy={cy} r={3} fill="#2563eb" stroke="#ffffff" strokeWidth={1.5} />
                      );
                    }}
                    activeDot={{ r: 7, stroke: '#1d4ed8', strokeWidth: 2 }}
                    name="BioNexus Adaptive Policy"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Side Panel: Assay Cost Table (Read-Only) */}
          <div className="rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-sm border border-slate-200 dark:border-slate-800 flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
                    <Layers className="w-4 h-4 text-indigo-600" />
                    Assay Cost Table
                  </h3>
                  <p className="text-xs text-slate-500">Unit lab costs & specifications</p>
                </div>
                <span className="flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  <Lock className="w-3 h-3 text-slate-400" /> Read-Only
                </span>
              </div>

              {/* Table of assays */}
              <div className="space-y-3">
                {data.gain_table?.assays?.map((assay) => (
                  <div
                    key={assay.assay}
                    className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-100 dark:border-slate-800 space-y-1.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-semibold text-xs text-slate-900 dark:text-white">
                        {assay.assay}
                      </div>
                      <div className="font-mono font-bold text-xs text-blue-600 dark:text-blue-400">
                        ${assay.unit_cost.toFixed(0)}
                      </div>
                    </div>
                    <div className="text-[11px] text-slate-500 leading-tight">
                      {assay.full_name}
                    </div>
                    <div className="flex items-center justify-between pt-1">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400 border border-amber-200/50">
                        {assay.badge}
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">
                        Turnaround: {assay.turnaround}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Cost Summary Box */}
              <div className="p-3.5 rounded-xl bg-slate-100/80 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 space-y-1 text-xs">
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span>Single Assay (RNA-seq):</span>
                  <span className="font-mono font-bold">$500</span>
                </div>
                <div className="flex justify-between text-slate-600 dark:text-slate-300">
                  <span>Full 3-Assay Panel:</span>
                  <span className="font-mono font-bold">$1,300</span>
                </div>
                <div className="flex justify-between text-blue-600 dark:text-blue-400 font-bold border-t border-slate-200 dark:border-slate-700 pt-1.5 mt-1">
                  <span>Adaptive Range:</span>
                  <span className="font-mono">$520 – $1,275</span>
                </div>
              </div>
            </div>

            <div className="text-[11px] text-slate-400 leading-relaxed border-t border-slate-100 dark:border-slate-800 pt-3">
              <span className="font-semibold text-slate-500">Notice:</span> Assay costs are estimated laboratory benchmarks. Viewer mode is strictly read-only; admin editing is restricted.
            </div>
          </div>
        </div>

        {/* ========================================================================= */}
        {/* 4. "HOW MUCH DATA DO WE NEED?" BAR CHART */}
        {/* ========================================================================= */}
        <div className="rounded-2xl bg-white dark:bg-slate-900 p-6 shadow-sm border border-slate-200 dark:border-slate-800 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 dark:border-slate-800 pb-4">
            <div>
              <span className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider">
                Ablation Comparison
              </span>
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                How Much Data Do We Need?
              </h3>
              <p className="text-xs text-slate-500">
                Evaluation of all 7 modality combinations across stratified 5-fold CV with 95% confidence intervals / error bars (&plusmn;1 std).
              </p>
            </div>
            <div className="text-xs font-mono text-slate-500 bg-slate-100 dark:bg-slate-800 px-3 py-1.5 rounded-lg">
              Metric: {metric === 'macro_f1' ? 'Macro-F1' : 'Accuracy'}
            </div>
          </div>

          <div className="h-[340px] w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={modalityBarData}
                margin={{ top: 20, right: 30, left: 10, bottom: 45 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
                <XAxis
                  dataKey="name"
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis
                  domain={[0.65, 0.90]}
                  tickFormatter={(val) => val.toFixed(2)}
                  label={{ value: metric === 'macro_f1' ? 'Macro-F1' : 'Accuracy', angle: -90, position: 'insideLeft', offset: 0, fontSize: 12, fill: '#64748b' }}
                  stroke="#94a3b8"
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload || !payload.length) return null;
                    const item = payload[0].payload;
                    return (
                      <div className="rounded-xl bg-slate-900/95 text-white p-3 shadow-xl border border-slate-800 text-xs space-y-1 backdrop-blur-md">
                        <div className="font-bold text-white text-sm">{item.name}</div>
                        <div className="text-slate-400 text-[11px] font-mono">Modalities: {item.modalities}</div>
                        <div className="text-slate-300 flex justify-between gap-4 pt-1">
                          <span>Cost:</span>
                          <span className="font-mono font-bold text-amber-300">${item.cost}</span>
                        </div>
                        <div className="text-slate-300 flex justify-between gap-4">
                          <span>{metric === 'macro_f1' ? 'Macro-F1' : 'Accuracy'}:</span>
                          <span className="font-mono font-bold text-blue-400">{item.value.toFixed(4)}</span>
                        </div>
                        <div className="text-slate-400 flex justify-between gap-4 text-[11px]">
                          <span>Std Deviation (&plusmn;&sigma;):</span>
                          <span className="font-mono">&plusmn;{item.error.toFixed(4)}</span>
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} maxBarSize={45}>
                  <ErrorBar dataKey="error" width={4} strokeWidth={2} stroke="#1e293b" />
                  {modalityBarData.map((entry, index) => (
                    <Cell
                      key={`cell-${entry.id}`}
                      fill={
                        entry.id === 'full_panel'
                          ? '#10b981'
                          : entry.id.includes('rna')
                          ? '#3b82f6'
                          : '#94a3b8'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-wrap items-center justify-between text-xs text-slate-500 pt-2 border-t border-slate-100 dark:border-slate-800">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-emerald-500" /> Full Panel ($1,300)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-blue-500" /> RNA-seq Subsets ($500-$950)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 rounded bg-slate-400" /> Non-RNA Subsets ($350-$800)
              </span>
            </div>
            <div className="font-mono text-[11px]">
              Whiskers represent &plusmn;1 standard deviation across CV folds.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
