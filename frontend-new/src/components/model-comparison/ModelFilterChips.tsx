import { Check, RotateCcw, Filter } from 'lucide-react';
import type { ModelComparisonItem } from '@/data/modelComparison';

interface ModelFilterChipsProps {
  allModels: ModelComparisonItem[];
  selectedModelIds: string[];
  onChange: (ids: string[]) => void;
}

export function ModelFilterChips({ allModels, selectedModelIds, onChange }: ModelFilterChipsProps) {
  const toggleModel = (id: string) => {
    // BioNexus is the focus model and cannot be toggled off
    if (id === 'bionexus') return;

    if (selectedModelIds.includes(id)) {
      if (selectedModelIds.length > 2) {
        onChange(selectedModelIds.filter(mId => mId !== id));
      }
    } else {
      onChange([...selectedModelIds, id]);
    }
  };

  const selectAll = () => {
    onChange(allModels.map(m => m.id));
  };

  const selectCoreBaselines = () => {
    onChange(['bionexus', 'mogonet', 'xgboost', 'single_omics']);
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 bg-slate-50/90 border border-slate-200/80 rounded-2xl">
      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-slate-500 shrink-0" />
        <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
          Filter Techniques:
        </span>
        <span className="text-xs text-slate-400">
          ({selectedModelIds.length} of {allModels.length} active)
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {allModels.map(model => {
          const isSelected = selectedModelIds.includes(model.id);
          const isFocus = model.id === 'bionexus';

          return (
            <button
              key={model.id}
              type="button"
              onClick={() => toggleModel(model.id)}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all
                ${isFocus
                  ? 'bg-blue-600 text-white shadow-xs hover:bg-blue-700 cursor-default'
                  : isSelected
                  ? 'bg-white text-slate-800 border border-slate-300 shadow-2xs hover:border-slate-400'
                  : 'bg-slate-200/60 text-slate-400 border border-transparent hover:bg-slate-200'
                }
              `}
              title={isFocus ? 'BioNexus is the benchmark focus model' : `Toggle ${model.name}`}
            >
              {isSelected && <Check className={`w-3.5 h-3.5 ${isFocus ? 'text-white' : 'text-blue-600'}`} />}
              <span>{model.name}</span>
              {isFocus && (
                <span className="text-[10px] bg-white/20 text-white px-1.5 py-0.2 rounded-full font-bold">
                  Focus
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={selectAll}
          className="text-xs font-semibold text-blue-600 hover:text-blue-800 transition-colors px-2 py-1 rounded-md hover:bg-blue-50"
        >
          Select All
        </button>
        <span className="text-slate-300">|</span>
        <button
          type="button"
          onClick={selectCoreBaselines}
          className="flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-slate-700 transition-colors px-2 py-1 rounded-md hover:bg-slate-200/60"
        >
          <RotateCcw className="w-3 h-3" />
          Core Only
        </button>
      </div>
    </div>
  );
}
