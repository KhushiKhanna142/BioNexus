import { useState, useMemo } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer, Tooltip,
} from 'recharts';
import type { ModelComparisonItem } from '@/data/modelComparison';
import { computeRadarSeries, RADAR_AXES } from '@/data/modelComparison';
import { Compass, Sparkles } from 'lucide-react';

interface ModelRadarChartProps {
  allModels: ModelComparisonItem[];
}

export function ModelRadarChart({ allModels }: ModelRadarChartProps) {
  // Competitor models (excluding BioNexus)
  const competitors = useMemo(() => {
    return allModels.filter(m => m.id !== 'bionexus');
  }, [allModels]);

  const [competitorId, setCompetitorId] = useState<string>('xgboost');

  const selectedCompetitor = useMemo(() => {
    return competitors.find(c => c.id === competitorId) || competitors[0];
  }, [competitors, competitorId]);

  const radarData = useMemo(() => {
    const rawSeries = computeRadarSeries(allModels);
    return rawSeries.map(item => ({
      axis: item.axis,
      bionexus: item['bionexus'] || 30,
      competitor: (selectedCompetitor && item[selectedCompetitor.id]) || 30,
    }));
  }, [allModels, selectedCompetitor]);

  return (
    <div className="bio-card p-6 border border-slate-200/90 shadow-sm space-y-4">
      {/* Header & Competitor Selector */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
            <Compass className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-slate-900 text-base flex items-center gap-1.5">
              <span>Multi-Axis Profile:</span>
              <span className="text-blue-600">BioNexus vs Competitor</span>
            </h3>
            <p className="text-xs text-slate-500">
              Normalized 0–100 scale across predictive power, biological interpretability, and latency.
            </p>
          </div>
        </div>

        {/* Competitor Dropdown */}
        <div className="flex items-center gap-2">
          <label htmlFor="competitor-select" className="text-xs font-semibold text-slate-500 shrink-0">
            Compare Against:
          </label>
          <select
            id="competitor-select"
            value={selectedCompetitor?.id}
            onChange={(e) => setCompetitorId(e.target.value)}
            className="text-xs font-semibold bg-white border border-slate-200 rounded-xl px-3 py-2 text-slate-800 shadow-2xs hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
          >
            {competitors.map(c => (
              <option key={c.id} value={c.id}>
                {c.name} ({c.family})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Radar Chart Container */}
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
            <PolarGrid stroke="#E2E8F0" />
            <PolarAngleAxis
              dataKey="axis"
              tick={{ fill: '#475569', fontSize: 11, fontWeight: 600 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              tick={{ fill: '#94A3B8', fontSize: 9 }}
              axisLine={false}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-slate-900 text-white text-xs rounded-xl p-3 shadow-xl border border-slate-700 space-y-1.5">
                    <div className="font-bold text-slate-100 border-b border-slate-800 pb-1">
                      {d.axis}
                    </div>
                    <div className="flex items-center justify-between gap-4 text-blue-300">
                      <span className="font-medium">BioNexus:</span>
                      <span className="font-mono font-bold text-white">{d.bionexus}/100</span>
                    </div>
                    <div className="flex items-center justify-between gap-4 text-slate-300">
                      <span className="font-medium">{selectedCompetitor?.name}:</span>
                      <span className="font-mono font-bold text-white">{d.competitor}/100</span>
                    </div>
                  </div>
                );
              }}
            />
            <Radar
              name="BioNexus (Graph Transformer)"
              dataKey="bionexus"
              stroke="#0B1B3F"
              fill="#0B1B3F"
              fillOpacity={0.35}
              strokeWidth={2.5}
            />
            <Radar
              name={selectedCompetitor?.name || 'Competitor'}
              dataKey="competitor"
              stroke="#64748B"
              fill="#94A3B8"
              fillOpacity={0.25}
              strokeWidth={2}
            />
            <Legend
              wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Insight Banner */}
      <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-xs text-slate-600 flex items-start gap-2.5">
        <Sparkles className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-bold text-slate-800">Tradeoff Takeaway: </span>
          BioNexus establishes an unmatched outer envelope in biological accuracy, causal interpretability, and cross-cohort resilience. Conversely, {selectedCompetitor?.name} achieves lower training cost and sub-millisecond inference due to its lighter architectural footprint.
        </div>
      </div>
    </div>
  );
}
