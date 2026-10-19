import { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import type { ModelComparisonItem, MetricDefinition } from '@/data/modelComparison';
import { formatMetricValue, computeMetricRankings } from '@/data/modelComparison';
import { MetricTooltip } from './MetricTooltip';
import { Crown, BarChart3 } from 'lucide-react';

interface GroupedBarChartProps {
  models: ModelComparisonItem[];
  availableMetrics: MetricDefinition[];
  selectedMetricId: string;
  onMetricChange: (metricId: string) => void;
}

export function GroupedBarChart({
  models,
  availableMetrics,
  selectedMetricId,
  onMetricChange,
}: GroupedBarChartProps) {
  const currentMetric = useMemo(() => {
    return availableMetrics.find(m => m.id === selectedMetricId) || availableMetrics[0];
  }, [availableMetrics, selectedMetricId]);

  const ranking = useMemo(() => {
    if (!currentMetric) return null;
    return computeMetricRankings(models, currentMetric);
  }, [models, currentMetric]);

  const chartData = useMemo(() => {
    if (!currentMetric) return [];

    return models
      .map(m => {
        const val = (m.metrics as any)[currentMetric.id];
        let numVal = typeof val === 'boolean' ? (val ? 1 : 0) : val;

        // If unit is ratio (e.g. 0.94), multiply by 100 for intuitive chart percentages
        if (typeof numVal === 'number' && currentMetric.unit === 'ratio') {
          numVal = Number((numVal * 100).toFixed(2));
        }

        return {
          id: m.id,
          name: m.name,
          family: m.family,
          rawVal: val,
          value: numVal,
          formatted: formatMetricValue(val, currentMetric.unit),
          isFocus: m.id === 'bionexus',
          isWinner: ranking ? ranking.isWinner(m.id) : false,
        };
      })
      .filter(d => d.rawVal !== null && d.rawVal !== undefined);
  }, [models, currentMetric, ranking]);

  if (!currentMetric) return null;

  return (
    <div className="bio-card p-6 border border-slate-200/90 shadow-sm space-y-4">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center">
            <BarChart3 className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5 font-bold text-slate-900 text-base">
              <span>Benchmark Metric:</span>
              <MetricTooltip metric={currentMetric}>
                <span className="text-blue-600 underline decoration-dotted underline-offset-4 cursor-help">
                  {currentMetric.label}
                </span>
              </MetricTooltip>
            </div>
            <p className="text-xs text-slate-500">
              {currentMetric.description}
            </p>
          </div>
        </div>

        {/* Dropdown to switch metric */}
        <div className="flex items-center gap-2">
          <label htmlFor="metric-select" className="text-xs font-semibold text-slate-500 shrink-0">
            Metric:
          </label>
          <select
            id="metric-select"
            value={currentMetric.id}
            onChange={(e) => onMetricChange(e.target.value)}
            className="text-xs font-semibold bg-white border border-slate-200 rounded-xl px-3 py-2 text-slate-800 shadow-2xs hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
          >
            {availableMetrics.map(m => (
              <option key={m.id} value={m.id}>
                {m.label} ({m.direction === 'higher' ? '↑ higher' : '↓ lower'})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Recharts Bar Chart */}
      <div className="h-80 w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 24, left: 10, bottom: 45 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
            <XAxis
              dataKey="name"
              stroke="#64748B"
              fontSize={11}
              interval={0}
              angle={-25}
              textAnchor="end"
              height={50}
              tickLine={false}
            />
            <YAxis
              stroke="#64748B"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => {
                if (currentMetric.unit === 'ratio') return `${v}%`;
                if (currentMetric.unit === 'pct') return `${v}%`;
                if (currentMetric.unit === 'pp') return `${v}pp`;
                if (currentMetric.unit === 'ms') return `${v}ms`;
                if (currentMetric.unit === 'min') return `${v}m`;
                if (currentMetric.unit === 'M') return `${v}M`;
                if (currentMetric.kind === 'boolean') return v === 1 ? 'Yes' : 'No';
                return v;
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-slate-900 text-white text-xs rounded-xl p-3 shadow-xl border border-slate-700 space-y-1">
                    <div className="flex items-center justify-between gap-3 font-bold text-slate-100">
                      <span>{data.name}</span>
                      {data.isWinner && (
                        <span className="flex items-center gap-1 text-[10px] text-amber-300 font-semibold bg-amber-500/20 px-1.5 py-0.5 rounded">
                          <Crown className="w-3 h-3" /> Best Performer
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Family: {data.family}
                    </div>
                    <div className="pt-1.5 border-t border-slate-800 flex items-center justify-between gap-4">
                      <span className="text-slate-300">{currentMetric.label}:</span>
                      <span className="font-mono font-bold text-blue-400 text-sm">{data.formatted}</span>
                    </div>
                  </div>
                );
              }}
            />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} maxBarSize={48}>
              {chartData.map((entry) => {
                // BioNexus in prominent navy (#0B1B3F), others in muted grey-blue (#64748B)
                const fillColor = entry.isFocus ? '#0B1B3F' : '#94A3B8';
                return <Cell key={`cell-${entry.id}`} fill={fillColor} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Chart legend & explanation footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100 text-xs">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded bg-[#0B1B3F]" />
            <span className="font-bold text-slate-900">BioNexus (Navy)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-3.5 rounded bg-[#94A3B8]" />
            <span className="text-slate-600">Alternative Baselines (Muted Grey-Blue)</span>
          </div>
        </div>
        <div className="text-slate-500 font-medium">
          Target: {currentMetric.direction === 'higher' ? 'Higher is better (maximize)' : 'Lower is better (minimize)'}
        </div>
      </div>
    </div>
  );
}
