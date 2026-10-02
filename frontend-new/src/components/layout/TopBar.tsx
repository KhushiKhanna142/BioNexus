import { Search, Bell, Activity } from 'lucide-react';
import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/api/client';
import type { Patient } from '@/lib/types';
import { useHealth } from '@/lib/health';

interface TopBarProps {
  title: string;
  subtitle?: string;
  onSelectPatient?: (patientId: string) => void;
}

export function TopBar({ title, subtitle, onSelectPatient }: TopBarProps) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [searchError, setSearchError] = useState<string | null>(null);
  const health = useHealth();
  const navigate = useNavigate();

  useEffect(() => {
    if (patients.length) return;
    api.getPatients().then(setPatients).catch(err => setSearchError(err.message));
  }, [patients.length]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setSearchOpen(prev => !prev);
    }
    if (e.key === 'Escape') setSearchOpen(false);
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const matches = patients.filter(p => p.id.toLowerCase().includes(searchQuery.toLowerCase())).slice(0, 10);

  const handleSelect = (id: string) => {
    if (onSelectPatient) {
      onSelectPatient(id);
    } else {
      navigate(`/patient-analysis?patient=${encodeURIComponent(id)}`);
    }
    setSearchOpen(false);
    setSearchQuery('');
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const exact = patients.find(p => p.id.toLowerCase() === searchQuery.trim().toLowerCase());
    if (exact) {
      handleSelect(exact.id);
    } else if (matches.length > 0) {
      handleSelect(matches[0].id);
    }
  };

  return (
    <>
      <header className="sticky top-0 z-40 bg-white/95 backdrop-blur-md border-b border-slate-100 px-6 h-18 flex items-center justify-between transition-all">
        {/* Left: Title & Subtitle */}
        <div>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">{title}</h1>
          {subtitle && (
            <p className="text-xs text-slate-500 font-medium mt-0.5">{subtitle}</p>
          )}
        </div>

        {/* Right: Search, Notifications, Avatar */}
        <div className="flex items-center gap-3">
          {/* Quick Search Input */}
          <div className="relative">
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2.5 px-3.5 py-2 bg-slate-50 hover:bg-slate-100/80 border border-slate-200/80 rounded-xl text-xs text-slate-400 hover:text-slate-600 transition-colors w-48 sm:w-56"
              aria-label="Search patients (Ctrl+K)"
            >
              <Search className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              <span className="truncate">Search patient ID...</span>
              <kbd className="hidden sm:inline-block ml-auto text-[10px] bg-white px-1.5 py-0.5 rounded border border-slate-200 text-slate-500 shadow-2xs">
                ⌘K
              </kbd>
            </button>
          </div>

          {/* Live Indicator Badge */}
          {health.mode === 'live' ? (
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200/60 text-[11px] font-medium text-emerald-700">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span>Live Model</span>
            </div>
          ) : (
            <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-600">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400" />
              <span>Model unavailable</span>
            </div>
          )}
        </div>
      </header>

      {/* Global Search Dialog Modal */}
      {searchOpen && (
        <div
          className="fixed inset-0 z-[100] bg-slate-900/30 backdrop-blur-xs flex items-start justify-center pt-[15vh] px-4"
          onClick={() => setSearchOpen(false)}
        >
          <div
            className="w-full max-w-lg bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden animate-in fade-in zoom-in-95 duration-150"
            onClick={e => e.stopPropagation()}
          >
            <form onSubmit={handleSearchSubmit}>
              <div className="flex items-center gap-3 px-4 py-3.5 border-b border-slate-100">
                <Search className="w-4.5 h-4.5 text-slate-400 shrink-0" />
                <input
                  autoFocus
                  type="text"
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  placeholder="Type a TCGA patient ID (e.g. TCGA-AR-A1AR)..."
                  className="flex-1 text-sm outline-none placeholder:text-slate-400 bg-transparent text-slate-800"
                  aria-label="Search patient ID"
                />
                <button
                  type="button"
                  onClick={() => setSearchOpen(false)}
                  className="text-xs text-slate-400 hover:text-slate-600 px-1.5 py-0.5 rounded bg-slate-100"
                >
                  ESC
                </button>
              </div>
            </form>
            <div className="max-h-72 overflow-y-auto p-2">
              {searchError && <p className="p-3 text-xs text-rose-500">{searchError}</p>}
              {matches.map(p => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => handleSelect(p.id)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs hover:bg-blue-50/70 hover:text-blue-600 text-slate-700 transition-colors"
                >
                  <span className="font-mono font-medium">{p.id}</span>
                  <span className="px-2 py-0.5 rounded-full bg-slate-100 text-[10px] font-semibold text-slate-600">
                    {p.subtype || p.true_pam50}
                  </span>
                </button>
              ))}
              {searchQuery && matches.length === 0 && !searchError && (
                <p className="p-4 text-center text-xs text-slate-400">
                  No matching cohort ID found for "{searchQuery}".
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
