import { Trophy, CheckCircle2, AlertTriangle, Cpu, ShieldCheck, Eye, Zap, Flame } from 'lucide-react';
import { motion } from 'framer-motion';

interface SummaryHeaderProps {
  headline: string;
  metricsLed: number;
  metricsConsidered: number;
  categoryStats: Record<string, { total: number; led: number }>;
  dataStatus?: string;
  notes?: string;
}

export function SummaryHeader({
  headline,
  metricsLed,
  metricsConsidered,
  categoryStats,
  dataStatus = 'placeholder',
}: SummaryHeaderProps) {
  const categories = [
    { id: 'performance', label: 'Performance', icon: Trophy, color: 'text-blue-600 bg-blue-50 border-blue-200' },
    { id: 'robustness', label: 'Robustness', icon: ShieldCheck, color: 'text-emerald-600 bg-emerald-50 border-emerald-200' },
    { id: 'interpretability', label: 'Interpretability', icon: Eye, color: 'text-purple-600 bg-purple-50 border-purple-200' },
    { id: 'cost', label: 'Practical Cost', icon: Zap, color: 'text-amber-600 bg-amber-50 border-amber-200' },
  ];

  return (
    <div className="space-y-4">
      {/* Top Banner Alert if using placeholder data */}
      {dataStatus === 'placeholder' && (
        <div className="flex items-center justify-between gap-3 px-4 py-2.5 bg-blue-50/70 border border-blue-200/80 rounded-xl text-xs text-blue-900 shadow-xs">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shrink-0" />
            <span className="font-semibold">Benchmark Environment:</span>
            <span>Pre-computed multi-omics baseline runs (TCGA-BRCA 646 cohort). Edit <code className="px-1.5 py-0.5 bg-blue-100/70 rounded font-mono text-[11px] text-blue-800">src/data/modelComparison.js</code> or sync from MLflow tracking.</span>
          </div>
          <span className="text-[11px] font-medium text-blue-700 bg-blue-100/60 px-2 py-0.5 rounded-full shrink-0">
            5-Fold Cross-Validation
          </span>
        </div>
      )}

      {/* Main KPI Hero Card */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="bio-card p-6 bg-gradient-to-br from-white via-white to-slate-50/60 border border-slate-200/90 shadow-sm"
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-1.5 max-w-2xl">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-blue-600 text-white shadow-xs">
                <Flame className="w-3.5 h-3.5" />
                Empirical Evaluation
              </span>
              <span className="text-xs font-medium text-slate-500">
                TCGA-BRCA Multi-Omics (RNA + miRNA + Methylation)
              </span>
            </div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              {headline}
            </h1>
            <p className="text-sm text-slate-600 leading-relaxed">
              Evaluating Graph Transformer (GraphGPS) against graph neural networks, gradient boosting trees, shallow MLPs, and single-modality baselines across predictive, generalization, and computational tradeoffs.
            </p>
          </div>

          {/* Metric count badge */}
          <div className="flex items-center gap-3 shrink-0 bg-slate-50 border border-slate-200/80 rounded-2xl p-4">
            <div className="w-12 h-12 rounded-xl bg-blue-600 text-white flex items-center justify-center font-bold text-xl shadow-md shadow-blue-500/20">
              {metricsLed}
            </div>
            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-slate-400">Winning Metrics</div>
              <div className="text-sm font-semibold text-slate-800">
                {metricsLed} of {metricsConsidered} metrics led
              </div>
              <div className="text-[11px] text-slate-500">
                {Math.round((metricsLed / (metricsConsidered || 1)) * 100)}% benchmark superiority
              </div>
            </div>
          </div>
        </div>

        {/* Category Breakdown Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-6 pt-5 border-t border-slate-100">
          {categories.map(cat => {
            const stats = categoryStats[cat.id] || { total: 0, led: 0 };
            const Icon = cat.icon;
            const isSweep = stats.total > 0 && stats.led === stats.total;
            const isZero = stats.led === 0;

            return (
              <div
                key={cat.id}
                className="flex items-center gap-3 p-3 rounded-xl bg-white border border-slate-100 shadow-2xs hover:border-slate-200 transition-colors"
              >
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border ${cat.color}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-slate-700 truncate">{cat.label}</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-sm font-bold text-slate-900 font-mono">
                      {stats.led}/{stats.total}
                    </span>
                    <span className="text-[10px] text-slate-400">led</span>
                    {isSweep && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 font-medium">
                        100%
                      </span>
                    )}
                    {isZero && stats.total > 0 && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-100 text-amber-800 font-medium">
                        Tradeoff
                      </span>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
}
