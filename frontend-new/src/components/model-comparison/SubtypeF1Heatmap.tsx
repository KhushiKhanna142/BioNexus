import { useState, useMemo } from 'react';
import type { ModelComparisonItem } from '@/data/modelComparison';
import { PAM50_SUBTYPES, MINORITY_SUBTYPES } from '@/data/modelComparison';
import { Layers, Grid, BarChart2, AlertCircle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

interface SubtypeF1HeatmapProps {
  models: ModelComparisonItem[];
}

export function SubtypeF1Heatmap({ models }: SubtypeF1HeatmapProps) {
  const [viewMode, setViewMode] = useState<'heatmap' | 'bars'>('heatmap');

  // Heatmap cell color generator based on F1 value [0.50, 1.00]
  const getCellBg = (f1: number) => {
    if (f1 >= 0.95) return 'bg-emerald-600 text-white font-bold';
    if (f1 >= 0.90) return 'bg-emerald-500 text-white font-bold';
    if (f1 >= 0.85) return 'bg-blue-500 text-white font-semibold';
    if (f1 >= 0.80) return 'bg-blue-400 text-white font-semibold';
    if (f1 >= 0.75) return 'bg-slate-300 text-slate-900 font-medium';
    if (f1 >= 0.70) return 'bg-amber-200 text-amber-950 font-medium';
    if (f1 >= 0.60) return 'bg-amber-300 text-amber-950 font-semibold';
    return 'bg-rose-400 text-white font-bold';
  };

  // Subtype colors matching the application's design system
  const SUBTYPE_THEMES: Record<string, { bg: string; text: string; barColor: string }> = {
    LumA: { bg: 'bg-blue-50', text: 'text-blue-700', barColor: '#4F6FE8' },
    LumB: { bg: 'bg-pink-50', text: 'text-pink-700', barColor: '#F47C9C' },
    Her2: { bg: 'bg-amber-50', text: 'text-amber-800', barColor: '#F5B942' },
    Basal: { bg: 'bg-purple-50', text: 'text-purple-700', barColor: '#8B6FE0' },
    Normal: { bg: 'bg-emerald-50', text: 'text-emerald-700', barColor: '#34C38F' },
  };

  // Data format for grouped bar chart view
  const barChartData = useMemo(() => {
    return models.map(m => {
      const entry: Record<string, string | number> = {
        name: m.name.split(' (')[0], // shorten name for bar label
      };
      PAM50_SUBTYPES.forEach(sub => {
        entry[sub] = Math.round((m.perClassF1[sub] || 0) * 100);
      });
      return entry;
    });
  }, [models]);

  return (
    <div className="bio-card p-6 border border-slate-200/90 shadow-sm space-y-4">
      {/* Header and View Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-pink-50 text-pink-600 flex items-center justify-center">
            <Layers className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-slate-900 text-base">
              PAM50 Subtype Breakdown: Per-Class F1 Score
            </h3>
            <p className="text-xs text-slate-500">
              Diagnostic matrix highlighting where alternative architectures collapse on rare minority subtypes.
            </p>
          </div>
        </div>

        {/* View Toggle */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-xl">
          <button
            type="button"
            onClick={() => setViewMode('heatmap')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              viewMode === 'heatmap' ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <Grid className="w-3.5 h-3.5" />
            Heatmap
          </button>
          <button
            type="button"
            onClick={() => setViewMode('bars')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              viewMode === 'bars' ? 'bg-white text-slate-900 shadow-2xs' : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            Grouped Bars
          </button>
        </div>
      </div>

      {/* Heatmap View */}
      {viewMode === 'heatmap' && (
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse" role="table">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="py-3 px-4 text-xs font-bold uppercase tracking-wider text-slate-400 bg-slate-50/50 rounded-tl-xl w-64">
                  Evaluated Model
                </th>
                {PAM50_SUBTYPES.map(sub => {
                  const isMinority = (MINORITY_SUBTYPES as readonly string[]).includes(sub);
                  return (
                    <th key={sub} className="py-3 px-3 text-center text-xs font-bold text-slate-700">
                      <div className="flex flex-col items-center">
                        <span className={`px-2 py-0.5 rounded-md text-xs font-bold ${SUBTYPE_THEMES[sub].bg} ${SUBTYPE_THEMES[sub].text}`}>
                          {sub}
                        </span>
                        {isMinority ? (
                          <span className="text-[10px] text-amber-600 font-semibold tracking-tight mt-0.5">
                            ★ Minority
                          </span>
                        ) : (
                          <span className="text-[10px] text-slate-400 mt-0.5">Majority</span>
                        )}
                      </div>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {models.map(m => {
                const isFocus = m.id === 'bionexus';
                return (
                  <tr
                    key={m.id}
                    className={`transition-colors ${
                      isFocus
                        ? 'bg-blue-50/60 font-semibold border-l-4 border-blue-600'
                        : 'hover:bg-slate-50/70'
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${isFocus ? 'font-bold text-blue-900' : 'text-slate-800'}`}>
                          {m.name}
                        </span>
                        {isFocus && (
                          <span className="text-[10px] bg-blue-600 text-white font-bold px-1.5 py-0.2 rounded-full">
                            Champion
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400">{m.family}</div>
                    </td>
                    {PAM50_SUBTYPES.map(sub => {
                      const f1 = m.perClassF1[sub];
                      const formatted = (f1 * 100).toFixed(1) + '%';
                      const cellStyle = getCellBg(f1);

                      return (
                        <td key={sub} className="py-2.5 px-3 text-center">
                          <div
                            className={`mx-auto w-16 py-1.5 rounded-lg text-xs font-mono shadow-2xs ${cellStyle}`}
                            title={`${m.name} ${sub} F1: ${formatted}`}
                          >
                            {formatted}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Grouped Bars View */}
      {viewMode === 'bars' && (
        <div className="h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={barChartData}
              margin={{ top: 20, right: 20, left: 10, bottom: 45 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
              <XAxis
                dataKey="name"
                stroke="#64748B"
                fontSize={11}
                interval={0}
                angle={-20}
                textAnchor="end"
                height={50}
                tickLine={false}
              />
              <YAxis
                domain={[40, 100]}
                stroke="#64748B"
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                formatter={(val, name) => [`${val}%`, `F1: ${name}`]}
                contentStyle={{ backgroundColor: '#0F172A', borderRadius: '12px', border: '1px solid #334155', color: '#FFF' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              {PAM50_SUBTYPES.map(sub => (
                <Bar
                  key={sub}
                  dataKey={sub}
                  fill={SUBTYPE_THEMES[sub].barColor}
                  radius={[4, 4, 0, 0]}
                  maxBarSize={14}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Minority Subtype Callout Notice */}
      <div className="p-3.5 bg-amber-50/70 border border-amber-200/80 rounded-xl flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 leading-relaxed">
          <span className="font-bold">Minority Subtype Resilience: </span>
          In breast cancer subtyping, <span className="font-semibold">Normal-like</span> and <span className="font-semibold">Her2</span> subtypes represent under 12% of TCGA samples. Flat baselines (Random Forest: 66.0%, Single-Omics: 54.0%) severely misclassify rare cases. BioNexus leverages cross-attention with epigenetic methylation layers to achieve <span className="font-bold text-emerald-800">86.0% Normal</span> and <span className="font-bold text-emerald-800">90.0% Her2</span> recall.
        </div>
      </div>
    </div>
  );
}
