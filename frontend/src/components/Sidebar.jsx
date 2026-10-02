import { useState, useMemo } from 'react'
import {
  Dna, Home, User, Layers, Share2, Sparkles, ShieldCheck,
  BarChart3, Scale, Activity, FlaskConical
} from 'lucide-react'

const NAV_GROUPS = [
  {
    label: 'Overview',
    items: [
      { key: 'dashboard', label: 'Dashboard', icon: Home, disabled: true },
      { key: 'explorer', label: 'Patient Explorer', icon: User },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { key: 'omics', label: 'Omics Explorer', icon: Layers, disabled: true },
      { key: 'graph', label: 'Graph Explorer', icon: Share2 },
    ],
  },
  {
    label: 'Interpretability',
    items: [
      { key: 'explainability', label: 'Explainability', icon: Sparkles, disabled: true },
      { key: 'causal', label: 'Causal Validation', icon: ShieldCheck, disabled: true },
    ],
  },
  {
    label: 'Evaluation',
    items: [
      { key: 'metrics', label: 'Model Performance', icon: BarChart3 },
      { key: 'comparison', label: 'Model Comparison', icon: Scale, disabled: true },
      { key: 'diagnostics', label: 'Diagnostics', icon: Activity, disabled: true },
    ],
  },
  {
    label: 'MLOps',
    items: [
      { key: 'experiments', label: 'Experiments', icon: FlaskConical, disabled: true },
    ],
  },
]

export default function Sidebar({ patients, selectedId, onSelect, view, onViewChange }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    if (!query) return patients
    const q = query.toLowerCase()
    return patients.filter(p => p.patient_id.toLowerCase().includes(q))
  }, [patients, query])

  return (
    <aside className="sidebar">
      <div className="brand-row">
        <Dna size={22} color="#3FBFAE" strokeWidth={1.6} />
        <div>
          <div className="brand">BioNexus</div>
          <div className="brand-sub">Multi-Omics Intelligence</div>
        </div>
      </div>

      <nav className="nav-groups">
        {NAV_GROUPS.map(group => (
          <div key={group.label} className="nav-group">
            <div className="nav-group-label">{group.label}</div>
            {group.items.map(item => {
              const Icon = item.icon
              const active = view === item.key
              return (
                <button
                  key={item.key}
                  className={`nav-item ${active ? 'active' : ''} ${item.disabled ? 'disabled' : ''}`}
                  onClick={() => !item.disabled && onViewChange(item.key)}
                  disabled={item.disabled}
                >
                  <Icon size={15} strokeWidth={1.7} />
                  <span>{item.label}</span>
                  {item.disabled && <span className="soon-tag">soon</span>}
                </button>
              )
            })}
          </div>
        ))}
      </nav>

      {(view === 'explorer' || view === 'graph') && (
        <div className="patient-panel">
          <input
            className="search-input"
            placeholder="Search patient ID"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <div className="patient-list">
            {filtered.map(p => (
              <button
                key={p.patient_id}
                className={`patient-row ${p.patient_id === selectedId ? 'selected' : ''}`}
                onClick={() => onSelect(p.patient_id)}
              >
                <span className="patient-id">{p.patient_id}</span>
                <span className="patient-subtype">{p.true_subtype}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="sidebar-footer">
        <div className="footer-title">TCGA-BRCA</div>
        <div className="footer-line">646 patients</div>
        <div className="footer-line">3 omics layers</div>
        <div className="footer-divider" />
        <div className="footer-line">Model: GraphGPS</div>
        <div className="footer-line status"><span className="status-dot" />Status: Ready</div>
      </div>
    </aside>
  )
}
