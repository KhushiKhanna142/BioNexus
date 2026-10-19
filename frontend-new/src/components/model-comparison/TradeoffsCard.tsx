import type { ModelComparisonItem } from '@/data/modelComparison';
import { computeModelInsights } from '@/data/modelComparison';
import { CheckCircle2, AlertCircle, ArrowUpRight, ArrowDownRight, Sparkles, Scale } from 'lucide-react';
import { useMemo } from 'react';

interface TradeoffsCardProps {
  models: ModelComparisonItem[];
}

export function TradeoffsCard({ models }: TradeoffsCardProps) {
  const { excels, lags } = useMemo(() => {
    return computeModelInsights(models, 'bionexus');
  }, [models]);

  return (
    <div className="bio-card p-6 border border-slate-200/90 shadow-sm space-y-6">
      <div className="flex items-center gap-2 pb-3 border-b border-slate-100">
        <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
          <Scale className="w-4 h-4" />
        </div>
        <div>
          <h3 className="font-bold text-slate-900 text-base">
            Architectural Balance: Where BioNexus Excels vs Where Baselines Win
          </h3>
          <p className="text-xs text-slate-500">
            Empirical advantages of graph transformer representations alongside an honest assessment of computational overheads.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Column 1: Where BioNexus Excels */}
        <div className="p-5 rounded-2xl bg-emerald-50/40 border border-emerald-200/70 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center">
                <CheckCircle2 className="w-4 h-4" />
              </div>
              <h4 className="font-bold text-emerald-950 text-sm">
                Where BioNexus Leads (Graph Transformer Advantage)
              </h4>
            </div>
            <span className="text-[11px] font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
              {excels.length} Benchmarks
            </span>
          </div>

          <div className="space-y-3">
            {excels.map((item) => (
              <div
                key={item.metric.id}
                className="p-3.5 rounded-xl bg-white border border-emerald-100 shadow-2xs space-y-1.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="font-semibold text-slate-900 text-xs flex items-center gap-1.5">
                    <ArrowUpRight className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>{item.metric.label}</span>
                  </div>
                  <span className="text-[11px] font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 shrink-0">
                    {item.marginText}
                  </span>
                </div>

                <div className="text-xs text-slate-600 flex items-center gap-2">
                  <span className="font-mono font-bold text-blue-700">{item.focusVal}</span>
                  <span className="text-slate-400">vs</span>
                  <span className="text-slate-500">Next best: {item.runnerUpVal} ({item.runnerUpName})</span>
                </div>

                <p className="text-[11px] text-slate-500 leading-relaxed pt-1 border-t border-slate-50">
                  <span className="font-semibold text-slate-700">Why it excels: </span>
                  {item.reason}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Column 2: Where Others Win */}
        <div className="p-5 rounded-2xl bg-amber-50/40 border border-amber-200/70 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-full bg-amber-500 text-white flex items-center justify-center">
                <AlertCircle className="w-4 h-4" />
              </div>
              <h4 className="font-bold text-amber-950 text-sm">
                Where Baselines Win (Honest Practical Overhead)
              </h4>
            </div>
            <span className="text-[11px] font-bold text-amber-800 bg-amber-100 px-2 py-0.5 rounded-full">
              {lags.length} Tradeoffs
            </span>
          </div>

          <div className="space-y-3">
            {lags.map((item) => (
              <div
                key={item.metric.id}
                className="p-3.5 rounded-xl bg-white border border-amber-100 shadow-2xs space-y-1.5"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="font-semibold text-slate-900 text-xs flex items-center gap-1.5">
                    <ArrowDownRight className="w-4 h-4 text-amber-600 shrink-0" />
                    <span>{item.metric.label}</span>
                  </div>
                  <span className="text-[11px] font-bold text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200 shrink-0">
                    {item.gapText}
                  </span>
                </div>

                <div className="text-xs text-slate-600 flex items-center gap-2">
                  <span className="text-slate-500">BioNexus: {item.focusVal}</span>
                  <span className="text-slate-400">vs</span>
                  <span className="font-mono font-bold text-amber-800">Best: {item.winnerVal} ({item.winnerName})</span>
                </div>

                <p className="text-[11px] text-slate-500 leading-relaxed pt-1 border-t border-slate-50">
                  <span className="font-semibold text-slate-700">Why baselines win: </span>
                  {item.reason}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
