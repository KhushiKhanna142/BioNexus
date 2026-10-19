import { Info } from 'lucide-react';
import { useState } from 'react';
import type { MetricDefinition } from '@/data/modelComparison';

interface MetricTooltipProps {
  metric: MetricDefinition;
  children?: React.ReactNode;
}

export function MetricTooltip({ metric, children }: MetricTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <span className="relative inline-flex items-center gap-1">
      {children || <span>{metric.label}</span>}
      <button
        type="button"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onFocus={() => setIsOpen(true)}
        onBlur={() => setIsOpen(false)}
        className="text-slate-400 hover:text-blue-600 focus:outline-none transition-colors p-0.5 rounded cursor-help"
        aria-label={`About ${metric.label}`}
      >
        <Info className="w-3.5 h-3.5" />
      </button>

      {isOpen && (
        <span
          role="tooltip"
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 p-3 bg-slate-900 text-white text-xs rounded-xl shadow-xl border border-slate-700 leading-relaxed pointer-events-none animate-in fade-in zoom-in-95 duration-150"
        >
          <span className="block font-semibold text-slate-100 mb-1 flex items-center justify-between">
            <span>{metric.label}</span>
            <span className="text-[10px] text-blue-300 font-mono uppercase">
              {metric.direction === 'higher' ? 'Higher is better' : 'Lower is better'}
            </span>
          </span>
          <span className="text-slate-300 block">{metric.description}</span>
          <span className="block mt-1.5 pt-1.5 border-t border-slate-800 text-[10px] text-slate-400">
            Category: {metric.category.toUpperCase()}
          </span>
        </span>
      )}
    </span>
  );
}
