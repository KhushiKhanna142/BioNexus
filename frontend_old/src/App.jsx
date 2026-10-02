import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import PredictionPanel from './components/PredictionPanel'
import MetricsDashboard from './components/MetricsDashboard'
import { api } from './api'

export default function App() {
  const [patients, setPatients] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [view, setView] = useState('explorer')
  const [error, setError] = useState(null)

  useEffect(() => {
    api.patients()
      .then(list => {
        setPatients(list)
        if (list.length) setSelectedId(list[0].patient_id)
      })
      .catch(e => setError(e.message))
  }, [])

  return (
    <div className="app-shell">
      <Sidebar
        patients={patients}
        selectedId={selectedId}
        onSelect={setSelectedId}
        view={view}
        onViewChange={setView}
      />
      <main className="main">
        <div className="section-label">
          {view === 'explorer' ? 'patient explorer' : 'model performance'}
        </div>
        {error && (
          <div className="empty-state">
            Couldn't reach the API at localhost:8000 — is the FastAPI server running?<br />
            ({error})
          </div>
        )}
        {!error && view === 'explorer' && <PredictionPanel patientId={selectedId} />}
        {!error && view === 'metrics' && <MetricsDashboard />}
      </main>
    </div>
  )
}
