import { useState, useMemo } from 'react'

export default function Sidebar({ patients, selectedId, onSelect, view, onViewChange }) {
  const [query, setQuery] = useState('')

  const filtered = useMemo(() => {
    if (!query) return patients
    const q = query.toLowerCase()
    return patients.filter(p => p.patient_id.toLowerCase().includes(q))
  }, [patients, query])

  return (
    <aside className="sidebar">
      <div>
        <div className="brand">BioNexus</div>
        <div className="brand-sub">multi-omics subtype classifier</div>
      </div>

      <div className="nav-tabs">
        <button className={`nav-tab ${view === 'explorer' ? 'active' : ''}`} onClick={() => onViewChange('explorer')}>
          Patient Explorer
        </button>
        <button className={`nav-tab ${view === 'metrics' ? 'active' : ''}`} onClick={() => onViewChange('metrics')}>
          Model Performance
        </button>
      </div>

      {view === 'explorer' && (
        <>
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
        </>
      )}
    </aside>
  )
}
