import { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { TabPills, CardSkeleton, Skeleton } from '@/components/ui';
import {
  MODELS_DATA,
  METRIC_DEFINITIONS,
  CATEGORIES,
  computeComparisonSummary,
  type ModelComparisonItem,
  type MetricDefinition,
} from '@/data/modelComparison';
import { SummaryHeader } from '@/components/model-comparison/SummaryHeader';
import { ModelFilterChips } from '@/components/model-comparison/ModelFilterChips';
import { GroupedBarChart } from '@/components/model-comparison/GroupedBarChart';
import { ModelRadarChart } from '@/components/model-comparison/ModelRadarChart';
import { SubtypeF1Heatmap } from '@/components/model-comparison/SubtypeF1Heatmap';
import { ComparisonTable } from '@/components/model-comparison/ComparisonTable';
import { TradeoffsCard } from '@/components/model-comparison/TradeoffsCard';
import { api } from '@/api/client';
import { AlertTriangle, Sparkles, Scale, RefreshCw } from 'lucide-react';

export function ModelComparisonPage() {
  const [activeCategory, setActiveCategory] = useState<string>('performance');
  const [models, setModels] = useState<ModelComparisonItem[]>(MODELS_DATA);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(() =>
    MODELS_DATA.map(m => m.id)
  );
  const [selectedMetricId, setSelectedMetricId] = useState<string>('accuracy');
  const [loading, setLoading] = useState<boolean>(false);
  const [dataStatus, setDataStatus] = useState<string>('measured');
  const [dataSource, setDataSource] = useState<'local' | 'api'>('local');

  // Dynamically fetch measured dashboard data from FastAPI backend
  useEffect(() => {
    let isMounted = true;
    api.getModelComparisonDashboard()
      .then((data: any) => {
        if (!isMounted) return;
        if (data && data.meta) {
          setDataStatus(data.meta.data_status || 'measured');
          if (Array.isArray(data.meta.models) && data.meta.models.length > 0) {
            const apiModels: ModelComparisonItem[] = data.meta.models.map((m: any) => {
              const local = MODELS_DATA.find(ld => ld.id === m.id);
              const hasApiMetrics = m.metrics && typeof m.metrics.accuracy === 'number';
              const hasApiClasses = m.per_class_f1 && Object.keys(m.per_class_f1).length > 0;

              return {
                id: m.id,
                name: m.name,
                family: m.family,
                highlighted: m.id === 'bionexus',
                metrics: hasApiMetrics ? {
                  accuracy: m.metrics.accuracy,
                  macroF1: m.metrics.macro_f1 ?? m.metrics.macroF1 ?? 0,
                  auc: m.metrics.auc_roc ?? m.metrics.auc ?? 0,
                  crossCohortDrop: m.metrics.cross_cohort_drop ?? null,
                  noiseRobustness: m.metrics.noise_robustness ?? null,
                  minorityF1: m.metrics.minority_f1 ?? m.metrics.minorityF1 ?? 0,
                  attentionExplanations: Boolean(m.metrics.attention_explanations),
                  causalAblation: Boolean(m.metrics.causal_ablation),
                  cosmicOverlapPct: m.metrics.cosmic_overlap_pct ?? null,
                  trainTimeMin: m.metrics.train_time_min ?? m.metrics.trainTimeMin ?? 0,
                  inferenceMs: m.metrics.inference_ms ?? m.metrics.inferenceMs ?? 0,
                  params: m.metrics.params_m ?? m.metrics.params ?? null,
                  gpuRequired: Boolean(m.metrics.gpu_required),
                } : (local?.metrics || {
                  accuracy: 0,
                  macroF1: 0,
                  auc: 0,
                  crossCohortDrop: null,
                  noiseRobustness: null,
                  minorityF1: 0,
                  attentionExplanations: false,
                  causalAblation: false,
                  cosmicOverlapPct: null,
                  trainTimeMin: 0,
                  inferenceMs: 0,
                  params: null,
                  gpuRequired: false,
                }),
                perClassF1: hasApiClasses ? {
                  LumA: m.per_class_f1.LumA ?? 0,
                  LumB: m.per_class_f1.LumB ?? 0,
                  Her2: m.per_class_f1.Her2 ?? 0,
                  Basal: m.per_class_f1.Basal ?? 0,
                  Normal: m.per_class_f1.Normal ?? 0,
                } : (local?.perClassF1 || {
                  LumA: 0,
                  LumB: 0,
                  Her2: 0,
                  Basal: 0,
                  Normal: 0,
                }),
              };
            });
            setModels(apiModels);
            setSelectedModelIds(prev =>
              prev.length > 0 && prev.every(id => apiModels.some(am => am.id === id))
                ? prev
                : apiModels.map(am => am.id)
            );
          }
          setDataSource('api');
        }
      })
      .catch(() => {
        setDataSource('local');
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Filter models based on user selection
  const activeModels = useMemo(() => {
    return models.filter(m => selectedModelIds.includes(m.id));
  }, [models, selectedModelIds]);

  // Metrics for currently selected category
  const categoryMetrics = useMemo(() => {
    return METRIC_DEFINITIONS.filter(m => m.category === activeCategory);
  }, [activeCategory]);

  // When category changes, ensure selectedMetricId belongs to that category
  useEffect(() => {
    const firstMetric = categoryMetrics[0];
    if (firstMetric && !categoryMetrics.some(m => m.id === selectedMetricId)) {
      setSelectedMetricId(firstMetric.id);
    }
  }, [activeCategory, categoryMetrics, selectedMetricId]);

  // Dynamically compute summary headline and counts
  const summary = useMemo(() => {
    return computeComparisonSummary(activeModels, 'bionexus');
  }, [activeModels]);

  const activeCategoryObj = useMemo(() => {
    return CATEGORIES.find(c => c.id === activeCategory) || CATEGORIES[0];
  }, [activeCategory]);

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      <TopBar
        title="Model Comparison"
        subtitle="Empirical evaluation: why graph transformers outperform traditional machine learning in multi-omics cancer subtyping."
      />

      <div className="p-6 space-y-6 max-w-7xl mx-auto">
        {loading ? (
          <div className="space-y-6">
            <CardSkeleton />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <CardSkeleton />
              <CardSkeleton />
            </div>
          </div>
        ) : (
          <>
            {/* 1. Header & One-line Summary Card */}
            <SummaryHeader
              headline={summary.headline}
              metricsLed={summary.metricsLed}
              metricsConsidered={summary.metricsConsidered}
              categoryStats={summary.categoryStats}
              dataStatus={dataStatus}
            />

            {/* 2. Model Filter Chips */}
            <ModelFilterChips
              allModels={models}
              selectedModelIds={selectedModelIds}
              onChange={setSelectedModelIds}
            />

            {/* 3. Category Tabs */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <TabPills
                tabs={CATEGORIES.map(c => ({
                  id: c.id,
                  label: c.label,
                  badge: `${summary.categoryStats[c.id]?.led || 0}/${summary.categoryStats[c.id]?.total || 0} led`,
                }))}
                activeTab={activeCategory}
                onChange={setActiveCategory}
              />
            </div>

            {/* 4. Grouped Bar Chart for the active category */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeCategory}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
              >
                <GroupedBarChart
                  models={activeModels}
                  availableMetrics={categoryMetrics}
                  selectedMetricId={selectedMetricId}
                  onMetricChange={setSelectedMetricId}
                />
              </motion.div>
            </AnimatePresence>

            {/* 5. Deep-Dive Section: Radar Chart & Subtype F1 Heatmap */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Radar Chart: BioNexus vs Competitor */}
              <ModelRadarChart allModels={activeModels} />

              {/* PAM50 Subtype F1 Heatmap */}
              <SubtypeF1Heatmap models={activeModels} />
            </div>

            {/* 6. Comprehensive Metric Comparison Table */}
            <ComparisonTable
              models={activeModels}
              metrics={categoryMetrics}
              activeCategoryLabel={activeCategoryObj.label}
            />

            {/* 7. "Where BioNexus Excels / Where Others Win" Two-Column Card */}
            <TradeoffsCard models={activeModels} />

            {/* Research Disclaimer */}
            <footer className="pt-4 pb-8 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500 border-t border-slate-200/80">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                <span>Research prototype, not for clinical use.</span>
              </div>
              <div className="flex items-center gap-3 text-[11px] text-slate-400">
                <span>TCGA-BRCA locked cohort (n=646)</span>
                <span>•</span>
                <span>PAM50 Subtype Classification</span>
                <span>•</span>
                <span>GraphGPS + Cross-Attention</span>
              </div>
            </footer>
          </>
        )}
      </div>
    </div>
  );
}
