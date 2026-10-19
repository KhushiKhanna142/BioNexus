import { useState, useMemo } from 'react';
import type { ModelComparisonItem, MetricDefinition } from '@/data/modelComparison';
import { formatMetricValue, computeMetricRankings } from '@/data/modelComparison';
import { MetricTooltip } from './MetricTooltip';
import { Crown, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

interface ComparisonTableProps {
  models: ModelComparisonItem[];
  metrics: MetricDefinition[];
  activeCategoryLabel: string;
}

export function ComparisonTable({ models, metrics, activeCategoryLabel }: ComparisonTableProps) {
  const [sortKey, setSortKey] = useState<string>('accuracy');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  // Compute best values and winners for each metric column
  const metricRankings = useMemo(() => {
    const map = new Map<string, ReturnType<typeof computeMetricRankings>>();
    metrics.forEach(m => {
      map.set(m.id, computeMetricRankings(models, m));
    });
    return map;
  }, [models, metrics]);

  // Handle column header click for sorting
  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      const metricDef = metrics.find(m => m.id === key);
      // Default to "best value first"
      setSortDir(metricDef?.direction === 'lower' ? 'asc' : 'desc');
    }
  };

  // Sort models according to current sortKey
  const sortedModels = useMemo(() => {
    const list = [...models];
    list.sort((a, b) => {
      if (sortKey === 'name') {
        return sortDir === 'asc'
          ? a.name.localeCompare(b.name)
          : b.name.localeCompare(a.name);
      }
      if (sortKey === 'family') {
        return sortDir === 'asc'
          ? a.family.localeCompare(b.family)
          : b.family.localeCompare(a.family);
      }

      const valA = (a.metrics as any)[sortKey];
      const valB = (b.metrics as any)[sortKey];

      if (valA === null || valA === undefined) return 1;
      if (valB === null || valB === undefined) return -1;

      const numA = typeof valA === 'boolean' ? (valA ? 1 : 0) : Number(valA);
      const numB = typeof valB === 'boolean' ? (valB ? 1 : 0) : Number(valB);

      return sortDir === 'asc' ? numA - numB : numB - numA;
    });
    return list;
  }, [models, sortKey, sortDir]);

  return (
    <div className="bio-card overflow-hidden border border-slate-200/90 shadow-sm">
      <div className="p-4 bg-slate-50/70 border-b border-slate-100 flex items-center justify-between">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Metric Matrix: <span className="text-slate-900 font-bold">{activeCategoryLabel}</span>
        </div>
        <div className="text-xs text-slate-400">
          Click any column header to sort models
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse" role="table">
          <thead>
            <tr className="border-b border-slate-200 bg-white">
              <th
                onClick={() => handleSort('name')}
                className="py-3 px-4 text-xs font-bold uppercase tracking-wider text-slate-500 hover:text-blue-600 cursor-pointer select-none w-64"
              >
                <div className="flex items-center gap-1.5">
                  <span>Architecture</span>
                  {sortKey === 'name' ? (
                    sortDir === 'asc' ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />
                  ) : (
                    <ArrowUpDown className="w-3.5 h-3.5 text-slate-300" />
                  )}
                </div>
              </th>

              {metrics.map(metric => {
                const isSorted = sortKey === metric.id;
                return (
                  <th
                    key={metric.id}
                    onClick={() => handleSort(metric.id)}
                    className="py-3 px-3.5 text-xs font-bold text-slate-700 hover:text-blue-600 cursor-pointer select-none whitespace-nowrap"
                  >
                    <div className="flex items-center gap-1">
                      <MetricTooltip metric={metric}>
                        <span className="border-b border-dotted border-slate-300">
                          {metric.label}
                        </span>
                      </MetricTooltip>
                      {isSorted ? (
                        sortDir === 'asc' ? <ArrowUp className="w-3.5 h-3.5 text-blue-600" /> : <ArrowDown className="w-3.5 h-3.5 text-blue-600" />
                      ) : (
                        <ArrowUpDown className="w-3.5 h-3.5 text-slate-300 opacity-60" />
                      )}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {sortedModels.map(model => {
              const isBioNexus = model.id === 'bionexus';

              return (
                <tr
                  key={model.id}
                  className={`transition-colors ${
                    isBioNexus
                      ? 'bg-blue-50/70 font-semibold border-l-4 border-blue-600'
                      : 'hover:bg-slate-50/60'
                  }`}
                >
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs ${isBioNexus ? 'font-bold text-blue-900' : 'text-slate-900'}`}>
                        {model.name}
                      </span>
                      {isBioNexus && (
                        <span className="text-[10px] bg-blue-600 text-white font-bold px-1.5 py-0.2 rounded-full">
                          BioNexus
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 font-normal">{model.family}</div>
                  </td>

                  {metrics.map(metric => {
                    const ranking = metricRankings.get(metric.id);
                    const isWinner = ranking ? ranking.isWinner(model.id) : false;
                    const rawVal = (model.metrics as any)[metric.id];
                    const formatted = formatMetricValue(rawVal, metric.unit);

                    return (
                      <td key={metric.id} className="py-3.5 px-3.5 whitespace-nowrap">
                        <div className="flex items-center gap-1.5 font-mono text-xs">
                          {isWinner ? (
                            <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300 shadow-2xs">
                              <Crown className="w-3 h-3 text-amber-500 fill-amber-500" />
                              <span>{formatted}</span>
                            </span>
                          ) : (
                            <span className={`${rawVal === null ? 'text-slate-300 italic' : isBioNexus ? 'text-slate-900 font-semibold' : 'text-slate-600'}`}>
                              {formatted}
                            </span>
                          )}
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
    </div>
  );
}
