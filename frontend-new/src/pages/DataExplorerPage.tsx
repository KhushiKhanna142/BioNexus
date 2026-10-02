import { useState, useEffect, useMemo } from 'react';
import { TopBar } from '@/components/layout/TopBar';
import { SectionCard } from '@/components/ui';
import { SUBTYPE_COLORS } from '@/lib/types';
import type { Patient } from '@/lib/types';
import { api } from '@/api/client';
import { Search, Download, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react';

const PAGE_SIZE = 20;

export function DataExplorerPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [sortField, setSortField] = useState<'id' | 'subtype' | 'confidence'>('id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    api.getPatients().then(data => { setPatients(data); setLoading(false); }).catch(err => { console.error(err); setLoading(false); });
  }, []);

  const filtered = useMemo(() => {
    let data = [...patients];
    if (search) {
      data = data.filter(p => p.id.toLowerCase().includes(search.toLowerCase()) ||
        (p.subtype && p.subtype.toLowerCase().includes(search.toLowerCase())));
    }
    data.sort((a, b) => {
      let cmp = 0;
      if (sortField === 'id') cmp = a.id.localeCompare(b.id);
      else if (sortField === 'subtype') cmp = (a.subtype || '').localeCompare(b.subtype || '');
      else if (sortField === 'confidence') cmp = (a.confidence || 0) - (b.confidence || 0);
      return sortDir === 'desc' ? -cmp : cmp;
    });
    return data;
  }, [patients, search, sortField, sortDir]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paged = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  const toggleSort = (field: typeof sortField) => {
    if (sortField === field) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortField(field); setSortDir('asc'); }
  };

  const exportCSV = () => {
    const headers = ['Patient ID', 'Subtype', 'Predicted Subtype', 'Confidence', 'RNA', 'miRNA', 'Methylation'];
    const rows = filtered.map(p => [
      p.id, p.subtype || '', p.predicted_subtype || '', (p.confidence || 0).toFixed(3),
      p.rna_available ? 'Yes' : 'No', p.mirna_available ? 'Yes' : 'No', p.methylation_available ? 'Yes' : 'No',
    ]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'bionexus_patients.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <TopBar title="Data Explorer" subtitle="Search, filter, and export patient data from the TCGA-BRCA cohort." />
      <div className="p-6">
        <SectionCard title="Patient Database" subtitle={`${filtered.length} patients found`}
          actions={<button onClick={exportCSV} className="btn-secondary text-xs"><Download className="w-3.5 h-3.5" /> Export CSV</button>}>
          <div className="flex items-center gap-2 px-3 py-2 bg-[var(--color-bg-primary)] mb-4">
            <Search className="w-4 h-4 text-[var(--color-text-muted)]" />
            <input value={search} onChange={e => { setSearch(e.target.value); setPage(0); }}
              placeholder="Search by patient ID or subtype..." className="flex-1 text-sm outline-none bg-transparent" />
          </div>
          {loading ? <div className="skeleton h-64" /> : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-[var(--color-text-muted)] border-b border-[var(--color-border)]">
                      <th className="text-left font-medium pb-2 pr-3">
                        <button onClick={() => toggleSort('id')} className="flex items-center gap-1 hover:text-[var(--color-text-primary)]">
                          Patient ID <ArrowUpDown className="w-3 h-3" />
                        </button>
                      </th>
                      <th className="text-left font-medium pb-2 pr-3">
                        <button onClick={() => toggleSort('subtype')} className="flex items-center gap-1 hover:text-[var(--color-text-primary)]">
                          Subtype <ArrowUpDown className="w-3 h-3" />
                        </button>
                      </th>
                      <th className="text-center font-medium pb-2 pr-3">RNA</th>
                      <th className="text-center font-medium pb-2 pr-3">miRNA</th>
                      <th className="text-center font-medium pb-2 pr-3">Methylation</th>
                      <th className="text-left font-medium pb-2 pr-3">Predicted</th>
                      <th className="text-right font-medium pb-2">
                        <button onClick={() => toggleSort('confidence')} className="flex items-center gap-1 ml-auto hover:text-[var(--color-text-primary)]">
                          Confidence <ArrowUpDown className="w-3 h-3" />
                        </button>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {paged.map(p => (
                      <tr key={p.id} className="border-b border-[var(--color-border-light)] hover:bg-[var(--color-bg-primary)] transition-colors">
                        <td className="py-2.5 pr-3 font-medium text-[var(--color-accent)]">{p.id}</td>
                        <td className="py-2.5 pr-3">
                          {p.subtype && (
                            <span className="px-2 py-0.5 text-xs font-medium" style={{ background: `${SUBTYPE_COLORS[p.subtype]}15`, color: SUBTYPE_COLORS[p.subtype] }}>
                              {p.subtype}
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 pr-3 text-center">{p.rna_available ? '✓' : '—'}</td>
                        <td className="py-2.5 pr-3 text-center">{p.mirna_available ? '✓' : '—'}</td>
                        <td className="py-2.5 pr-3 text-center">{p.methylation_available ? '✓' : '—'}</td>
                        <td className="py-2.5 pr-3">
                          {p.predicted_subtype && (
                            <span className="text-xs font-medium" style={{ color: SUBTYPE_COLORS[p.predicted_subtype] }}>{p.predicted_subtype}</span>
                          )}
                        </td>
                        <td className="py-2.5 text-right tabular-nums text-xs">{p.confidence?.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between mt-4 pt-3 border-t border-[var(--color-border)]">
                <p className="text-xs text-[var(--color-text-muted)]">
                  Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
                </p>
                <div className="flex items-center gap-1">
                  <button onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}
                    className="p-1.5 hover:bg-[var(--color-bg-primary)] disabled:opacity-30"><ChevronLeft className="w-4 h-4" /></button>
                  <span className="text-xs text-[var(--color-text-muted)] px-2">{page + 1} / {totalPages}</span>
                  <button onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page >= totalPages - 1}
                    className="p-1.5 hover:bg-[var(--color-bg-primary)] disabled:opacity-30"><ChevronRight className="w-4 h-4" /></button>
                </div>
              </div>
            </>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
