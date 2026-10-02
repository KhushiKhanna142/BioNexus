import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard, ErrorState } from '@/components/ui';
import { api } from '@/api/client';
import { ArrowUpDown, ExternalLink, ChevronDown, ChevronUp, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

export function BiomarkersPage() {
  const [topK, setTopK] = useState(10);
  const [sortBy, setSortBy] = useState<'importance' | 'gene'>('importance');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [showCausal, setShowCausal] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [layer, setLayer] = useState<'rna' | 'mirna' | 'meth'>('rna');
  const [patients, setPatients] = useState<{ id: string }[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [fusionNote, setFusionNote] = useState<string | null>(null);
  const [biomarkers, setBiomarkers] = useState<{ gene: string; importance: number; effect: string }[]>([]);
  const [causal, setCausal] = useState<{
    patients_larger_impact: number; total_patients: number; p_value: number;
    attributed_impact_median: number; random_impact_median: number; fold_difference: number; wording?: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getPatients().then(p => {
      setPatients(p);
      if (p.length) setPatientId(p[0].id);
    }).catch(err => setError(err.message));
  }, []);

  useEffect(() => {
    if (!patientId) return;
    setLoading(true);
    setError(null);
    api.getExplanation(patientId, 20, layer)
      .then(expl => setBiomarkers(expl.biomarkers))
      .catch(err => { setBiomarkers([]); setError(err.message); })
      .finally(() => setLoading(false));
  }, [patientId, layer]);

  useEffect(() => {
    api.getCausalValidation().then(setCausal).catch(() => setCausal(null));
    api.getFusionDiagnosis().then(d => setFusionNote(d.observation || null)).catch(() => setFusionNote(null));
  }, []);

  const sorted = useMemo(() => {
    const data = [...biomarkers].slice(0, topK);
    if (sortBy === 'gene') {
      data.sort((a, b) => sortDir === 'asc' ? a.gene.localeCompare(b.gene) : b.gene.localeCompare(a.gene));
    } else {
      data.sort((a, b) => sortDir === 'asc' ? a.importance - b.importance : b.importance - a.importance);
    }
    return data;
  }, [biomarkers, topK, sortBy, sortDir]);

  const toggleSort = (field: 'importance' | 'gene') => {
    if (sortBy === field) { setSortDir(d => d === 'asc' ? 'desc' : 'asc'); }
    else { setSortBy(field); setSortDir('desc'); }
  };

  return (
    <div>
      <TopBar title="Key Biomarkers" subtitle="Feature importance analysis via Integrated Gradients attribution." />
      <div className="p-6 space-y-6">
        <SectionCard title="Biomarker Importance"
          actions={
            <div className="flex items-center gap-3 flex-wrap">
              <select value={patientId} onChange={e => setPatientId(e.target.value)} className="text-xs border px-2 py-1 bg-transparent">
                {patients.map(p => <option key={p.id} value={p.id}>{p.id}</option>)}
              </select>
              <select value={layer} onChange={e => setLayer(e.target.value as 'rna' | 'mirna' | 'meth')} className="text-xs border px-2 py-1 bg-transparent">
                <option value="rna">RNA</option>
                <option value="mirna">miRNA</option>
                <option value="meth">Methylation</option>
              </select>
              <label className="text-xs text-[var(--color-text-muted)]">
                Top-K: <input type="range" min={5} max={20} value={topK} onChange={e => setTopK(Number(e.target.value))} className="ml-2 w-20 accent-[var(--color-accent)]" />
                <span className="ml-1 font-medium">{topK}</span>
              </label>
            </div>
          }>
          {error && <ErrorState error={error} />}
          {loading ? <div className="skeleton h-64" /> : (
            <>
              <p className="text-sm text-[var(--color-text-muted)] mb-4">
                These features had the highest |attribution| for the predicted class in layer <strong>{layer}</strong> (Integrated Gradients, 50 steps, precomputed).
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-[var(--color-text-muted)] border-b border-[var(--color-border)]">
                      <th className="text-left font-medium pb-3 pr-3">
                        <button onClick={() => toggleSort('gene')} className="flex items-center gap-1 hover:text-[var(--color-text-primary)]">Gene <ArrowUpDown className="w-3 h-3" /></button>
                      </th>
                      <th className="text-left font-medium pb-3 pr-3">
                        <button onClick={() => toggleSort('importance')} className="flex items-center gap-1 hover:text-[var(--color-text-primary)]">Importance Score <ArrowUpDown className="w-3 h-3" /></button>
                      </th>
                      <th className="text-right font-medium pb-3">Effect on Prediction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((b, i) => (
                      <motion.tr key={b.gene} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.03 }}
                        className="border-b border-[var(--color-border-light)] hover:bg-[var(--color-bg-primary)] transition-colors">
                        <td className="py-3 pr-3">
                          <a href={`https://www.genecards.org/cgi-bin/carddisp.pl?gene=${b.gene}`} target="_blank" rel="noopener noreferrer"
                            className="font-medium text-[var(--color-accent)] hover:underline inline-flex items-center gap-1">
                            {b.gene} <ExternalLink className="w-3 h-3" />
                          </a>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="flex items-center gap-3">
                            <span className="text-xs tabular-nums w-10">{b.importance.toFixed(2)}</span>
                            <div className="flex-1 h-2.5 bg-[var(--color-bg-primary)] overflow-hidden max-w-[200px]">
                              <motion.div initial={{ width: 0 }} animate={{ width: `${b.importance * 100}%` }}
                                transition={{ duration: 0.6, delay: i * 0.05 }} className="h-full bg-[var(--color-accent)]" />
                            </div>
                          </div>
                        </td>
                        <td className="py-3 text-right">
                          <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-1 ${
                            b.effect === 'positive' ? 'bg-green-50 text-[var(--color-success)]' : 'bg-red-50 text-[var(--color-error)]'
                          }`}>
                            {b.effect === 'positive' ? '↑' : '↓'} {b.effect === 'positive' ? 'Positive' : 'Negative'}
                          </span>
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </SectionCard>

        {causal && (
          <SectionCard title="Causal Validation" subtitle="Precomputed ablation: attributed vs random genes (logit impact)">
            <p className="text-sm text-[var(--color-text-muted)] mb-4">
              We ablated the top-20 attributed genes vs. 20 random genes (precomputed; not live).
              Attribution-flagged genes changed the model's logit more than random genes in{' '}
              {causal.patients_larger_impact}/{causal.total_patients} tested patients
              (Wilcoxon p&lt;{causal.p_value}). This does not prove biological causation.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-4">
              <div className="bio-card p-4 text-center">
                <p className="text-2xl font-bold text-[var(--color-navy)]">{causal.patients_larger_impact}/{causal.total_patients}</p>
                <p className="text-xs text-[var(--color-text-muted)]">Patients with larger attributed impact</p>
                <p className="text-lg font-bold text-[var(--color-accent)]">{((causal.patients_larger_impact / causal.total_patients) * 100).toFixed(0)}%</p>
              </div>
              <div className="bio-card p-4 text-center">
                <p className="text-2xl font-bold text-[var(--color-navy)]">{causal.fold_difference}×</p>
                <p className="text-xs text-[var(--color-text-muted)]">Median impact ratio</p>
                <p className="text-xs text-[var(--color-text-muted)]">({causal.attributed_impact_median} vs {causal.random_impact_median})</p>
              </div>
              <div className="bio-card p-4 text-center">
                <p className="text-2xl font-bold text-[var(--color-success)]">p &lt; {causal.p_value}</p>
                <p className="text-xs text-[var(--color-text-muted)]">Wilcoxon signed-rank test</p>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={[
                { name: 'Attributed Genes', impact: causal.attributed_impact_median, fill: 'var(--color-accent)' },
                { name: 'Random Genes', impact: causal.random_impact_median, fill: 'var(--color-text-muted)' },
              ]} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-light)" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'Median Impact', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip contentStyle={{ borderRadius: 0, fontSize: 12 }} />
                <Bar dataKey="impact" radius={0} />
              </BarChart>
            </ResponsiveContainer>
            <button onClick={() => setShowCausal(!showCausal)} className="flex items-center gap-1 text-xs text-[var(--color-accent)] mt-3 hover:underline">
              {showCausal ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />} What does this mean?
            </button>
            {showCausal && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mt-2 p-3 bg-[var(--color-bg-primary)]">
                <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
                  <strong>In plain English:</strong> When we removed the genes our model flagged as important,
                  the model's predictions changed significantly more than when we removed random genes.
                  This happened in {causal.patients_larger_impact} out of {causal.total_patients} patients tested.
                  The impact was about {causal.fold_difference}× larger for attributed genes,
                  This does not prove biological causation.
                </p>
              </motion.div>
            )}
            {fusionNote && (
              <p className="text-xs text-[var(--color-text-muted)] mt-3">Observation (not a proof): {fusionNote}</p>
            )}
          </SectionCard>
        )}
      </div>
    </div>
  );
}
