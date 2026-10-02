import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard, TabPills, GlossaryTerm, ErrorState } from '@/components/ui';
import { SUBTYPE_COLORS, SUBTYPES } from '@/lib/types';
import { api } from '@/api/client';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line,
} from 'recharts';
import { ArrowDown, Info, RefreshCw, Activity } from 'lucide-react';

export function CohortAnalysisPage() {
  const [activeTab, setActiveTab] = useState<'distribution' | 'embeddings' | 'survival'>('distribution');
  const [cohort, setCohort] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Embeddings Space & Method State
  const [embSpace, setEmbSpace] = useState<'fused' | 'rna' | 'mirna' | 'meth'>('fused');
  const [embMethod, setEmbMethod] = useState<'pca' | 'umap'>('umap');
  const [embeddingsData, setEmbeddingsData] = useState<{
    points: { patient_id: string; true_pam50: string; x: number; y: number }[];
    explained_variance?: number;
  } | null>(null);
  const [loadingEmbeddings, setLoadingEmbeddings] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState<{ patient_id: string; subtype: string; x: number; y: number } | null>(null);

  // Survival Curves State
  const [survivalData, setSurvivalData] = useState<{
    subtypes: Record<string, { timeline: number[]; survival_probability: number[]; median_survival: number | null; n_patients: number; n_events: number }>;
    logrank_p_luma_vs_basal?: number;
  } | null>(null);
  const [loadingSurvival, setLoadingSurvival] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    api.getCohort()
      .then(setCohort)
      .catch((err) => setError(err.message || 'Failed to load cohort stats'))
      .finally(() => setLoading(false));
  }, []);

  // Fetch embeddings when tab or options change
  useEffect(() => {
    if (activeTab !== 'embeddings') return;
    setLoadingEmbeddings(true);
    api.getEmbeddings(embSpace, embMethod)
      .then(setEmbeddingsData)
      .catch(console.error)
      .finally(() => setLoadingEmbeddings(false));
  }, [activeTab, embSpace, embMethod]);

  // Fetch survival data when tab changes
  useEffect(() => {
    if (activeTab !== 'survival') return;
    setLoadingSurvival(true);
    api.getSurvival('true')
      .then(setSurvivalData)
      .catch(console.error)
      .finally(() => setLoadingSurvival(false));
  }, [activeTab]);

  const dist = cohort?.subtype_distribution || {};
  const distData = SUBTYPES.map((s) => ({ name: s, count: dist[s] || 0, fill: SUBTYPE_COLORS[s] }));
  const funnel = cohort?.funnel || [];

  // Embedding bounds calculation
  const bounds = useMemo(() => {
    if (!embeddingsData?.points?.length) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    embeddingsData.points.forEach((p) => {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    });
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    return { minX, maxX, minY, maxY, spanX, spanY };
  }, [embeddingsData]);

  // Combined Survival Timeline Data for Recharts
  const survivalChartData = useMemo(() => {
    if (!survivalData?.subtypes) return [];
    const timePoints = new Set<number>();
    Object.values(survivalData.subtypes).forEach((sub) => {
      sub.timeline.forEach((t) => timePoints.add(t));
    });
    const sortedTimes = Array.from(timePoints).sort((a, b) => a - b);

    // Build step curves
    return sortedTimes.map((time) => {
      const entry: Record<string, number> = { time };
      Object.entries(survivalData.subtypes).forEach(([st, data]) => {
        let prob = 1.0;
        for (let i = 0; i < data.timeline.length; i++) {
          if (data.timeline[i] <= time) {
            prob = data.survival_probability[i];
          } else {
            break;
          }
        }
        entry[st] = prob;
      });
      return entry;
    });
  }, [survivalData]);

  return (
    <div>
      <TopBar title="Cohort Analysis" subtitle="Dataset distributions, multi-omics embeddings, and survival outcomes." />
      <div className="p-6 space-y-6">
        {error && <ErrorState error={error} />}

        <SectionCard title="Cohort Explorations">
          <TabPills
            tabs={[
              { id: 'distribution', label: 'Subtype & Funnel Distribution' },
              { id: 'embeddings', label: 'Multi-Omics Embeddings (PCA / UMAP)' },
              { id: 'survival', label: 'Kaplan-Meier Survival Analysis' },
            ]}
            activeTab={activeTab}
            onChange={(id) => setActiveTab(id as any)}
          />

          <div className="mt-6">
            {loading ? (
              <div className="skeleton h-[350px]" />
            ) : (
              <>
                {/* TAB 1: Subtype Distribution & Filtering Pipeline */}
                {activeTab === 'distribution' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
                    <div>
                      <p className="text-xs text-slate-500 mb-4">
                        PAM50 Subtype Distribution within locked cohort (n = {cohort?.total_patients || 646})
                      </p>
                      <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={distData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                          <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748B' }} />
                          <YAxis tick={{ fontSize: 12, fill: '#94A3B8' }} />
                          <Tooltip contentStyle={{ borderRadius: 12, border: '1px solid #E2E8F0', fontSize: 12 }} />
                          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                            {distData.map((entry, i) => (
                              <Cell key={i} fill={entry.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Funnel Pipeline */}
                    {funnel.length > 0 && (
                      <div className="pt-4 border-t border-slate-100">
                        <h4 className="text-xs font-bold text-slate-900 mb-1">Cohort Selection Funnel</h4>
                        <p className="text-[11px] text-slate-400 mb-4">
                          Three-modality intersection filtering from TCGA-BRCA raw data
                        </p>
                        <div className="flex flex-col items-center gap-2">
                          {funnel.map((step: any, i: number) => (
                            <div key={i} className="w-full max-w-md">
                              <div
                                className="bio-card p-3 text-center border border-slate-200"
                                style={{ width: `${100 - i * 12}%`, margin: '0 auto' }}
                              >
                                <p className="text-xl font-bold text-slate-900">{step.count.toLocaleString()}</p>
                                <p className="text-xs text-slate-600 font-medium">{step.label}</p>
                                <p className="text-[10px] text-slate-400 mt-0.5">{step.description}</p>
                              </div>
                              {i < funnel.length - 1 && (
                                <div className="flex justify-center my-1">
                                  <ArrowDown className="w-3.5 h-3.5 text-slate-400" />
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* TAB 2: Multi-Omics Embeddings (PCA / UMAP) */}
                {activeTab === 'embeddings' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-50 rounded-xl border border-slate-200">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-600">Representation:</span>
                        <div className="flex items-center gap-1 bg-white p-1 rounded-lg border border-slate-200">
                          {(['fused', 'rna', 'mirna', 'meth'] as const).map((space) => (
                            <button
                              key={space}
                              type="button"
                              onClick={() => setEmbSpace(space)}
                              className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
                                embSpace === space ? 'bg-[#0B1B3F] text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
                              }`}
                            >
                              {space === 'fused' ? 'Fused Embedding' : space.toUpperCase()}
                            </button>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-600">Method:</span>
                        <div className="flex items-center gap-1 bg-white p-1 rounded-lg border border-slate-200">
                          {(['umap', 'pca'] as const).map((method) => (
                            <button
                              key={method}
                              type="button"
                              onClick={() => setEmbMethod(method)}
                              className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-all ${
                                embMethod === method ? 'bg-blue-600 text-white shadow-2xs' : 'text-slate-600 hover:text-slate-900'
                              }`}
                            >
                              {method.toUpperCase()}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <p className="text-xs text-slate-500">
                      Hover over any patient point to view ID and subtype. Click to open Patient Analysis.
                      {embeddingsData?.explained_variance != null && (
                        <span className="ml-2 font-mono font-bold text-blue-600">
                          (Top-2 components explain {(embeddingsData.explained_variance * 100).toFixed(1)}% variance)
                        </span>
                      )}
                    </p>

                    <div className="relative h-[420px] w-full bg-slate-50 rounded-xl border border-slate-200 overflow-hidden flex items-center justify-center">
                      {loadingEmbeddings ? (
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Computing {embMethod.toUpperCase()} projection...
                        </div>
                      ) : bounds && embeddingsData?.points ? (
                        <svg className="w-full h-full p-4" viewBox="0 0 100 100">
                          {embeddingsData.points.map((pt) => {
                            const cx = ((pt.x - bounds.minX) / bounds.spanX) * 90 + 5;
                            const cy = ((pt.y - bounds.minY) / bounds.spanY) * 90 + 5;
                            const isHovered = hoveredPoint?.patient_id === pt.patient_id;
                            return (
                              <circle
                                key={pt.patient_id}
                                cx={cx}
                                cy={cy}
                                r={isHovered ? 4.5 : 2}
                                fill={(SUBTYPE_COLORS as Record<string, string>)[pt.true_pam50] || '#94A3B8'}
                                opacity={isHovered ? 1 : 0.8}
                                stroke={isHovered ? '#0F172A' : 'none'}
                                strokeWidth={isHovered ? 1.5 : 0}
                                className="cursor-pointer transition-all duration-100"
                                onMouseEnter={(e) => {
                                  const rect = (e.target as SVGElement).getBoundingClientRect();
                                  setHoveredPoint({
                                    patient_id: pt.patient_id,
                                    subtype: pt.true_pam50,
                                    x: rect.left,
                                    y: rect.top,
                                  });
                                }}
                                onMouseLeave={() => setHoveredPoint(null)}
                                onClick={() => navigate(`/patient-analysis?patient=${encodeURIComponent(pt.patient_id)}`)}
                              />
                            );
                          })}
                        </svg>
                      ) : (
                        <p className="text-xs text-slate-400">No embedding data available.</p>
                      )}

                      {/* Tooltip Overlay */}
                      {hoveredPoint && (
                        <div className="absolute top-3 left-3 bg-slate-900 text-white text-xs px-3 py-1.5 rounded-lg shadow-lg pointer-events-none font-mono">
                          <span className="font-bold">{hoveredPoint.patient_id}</span> ({hoveredPoint.subtype})
                        </div>
                      )}

                      {/* Subtype Legend Overlay */}
                      <div className="absolute bottom-3 right-3 bg-white/95 backdrop-blur-xs p-2 rounded-xl border border-slate-200 shadow-2xs flex flex-wrap gap-3 text-xs">
                        {Object.entries(SUBTYPE_COLORS).map(([st, color]) => (
                          <div key={st} className="flex items-center gap-1.5">
                            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                            <span className="font-semibold text-slate-700">{st}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* TAB 3: Kaplan-Meier Survival Analysis */}
                {activeTab === 'survival' && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    <div className="p-3 bg-blue-50 border border-blue-200 rounded-xl flex items-center justify-between text-xs text-blue-900">
                      <div>
                        <strong>Kaplan-Meier Overall Survival Curves:</strong> Stratified by PAM50 subtype from TCGA clinical records.
                      </div>
                      {survivalData?.logrank_p_luma_vs_basal != null && (
                        <div className="font-mono font-bold bg-white px-2 py-1 rounded border border-blue-200">
                          LumA vs Basal log-rank p = {survivalData.logrank_p_luma_vs_basal < 0.001 ? '< 0.001' : survivalData.logrank_p_luma_vs_basal.toFixed(4)}
                        </div>
                      )}
                    </div>

                    {loadingSurvival ? (
                      <div className="h-64 flex items-center justify-center text-xs text-slate-400">
                        <RefreshCw className="w-4 h-4 animate-spin mr-2" /> Computing Kaplan-Meier survival estimators...
                      </div>
                    ) : survivalChartData.length > 0 ? (
                      <div className="h-[360px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={survivalChartData} margin={{ top: 10, right: 30, bottom: 20, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                            <XAxis
                              dataKey="time"
                              tick={{ fontSize: 11, fill: '#64748B' }}
                              label={{ value: 'Overall Survival Time (Months)', position: 'insideBottom', offset: -10, fontSize: 12 }}
                            />
                            <YAxis
                              domain={[0, 1]}
                              tick={{ fontSize: 11, fill: '#64748B' }}
                              label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft', fontSize: 12 }}
                            />
                            <Tooltip
                              formatter={(value: any) => [`${(Number(value) * 100).toFixed(1)}%`, 'Probability']}
                              labelFormatter={(label) => `Month ${Number(label).toFixed(0)}`}
                              contentStyle={{ borderRadius: 12, border: '1px solid #E2E8F0', fontSize: 12 }}
                            />
                            {SUBTYPES.map((st) => (
                              <Line
                                key={st}
                                type="stepAfter"
                                dataKey={st}
                                stroke={SUBTYPE_COLORS[st]}
                                strokeWidth={2}
                                dot={false}
                                name={st}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400 text-center py-12">No survival data available.</p>
                    )}
                  </motion.div>
                )}
              </>
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
