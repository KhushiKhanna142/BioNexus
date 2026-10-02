import { useEffect, useState } from 'react'
import { api } from '../api'
import PatientGraphExplorer from './PatientGraphExplorer'

const SUBTYPE_ORDER = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
const LAYER_COLORS = { rna: '#3FBFAE', mirna: '#D97D66', meth: '#7B8FA6' }

export default function PredictionPanel({ patientId, onSelectPatient }) {
  const [prediction, setPrediction] = useState(null)
  const [genes, setGenes] = useState(null)
  const [layer, setLayer] = useState('rna')
  const [loadingGenes, setLoadingGenes] = useState(false)

  useEffect(() => {
    if (!patientId) return
    setPrediction(null)
    api.predict(patientId).then(setPrediction).catch(console.error)
  }, [patientId])

  useEffect(() => {
    if (!patientId) return
    setLoadingGenes(true)
    api.explain(patientId, layer, 10)
      .then(setGenes)
      .catch(console.error)
      .finally(() => setLoadingGenes(false))
  }, [patientId, layer])

  if (!patientId) {
    return <div className="empty-state">Select a patient from the list to view their prediction.</div>
  }
  if (!prediction) {
    return <div className="loading">Loading prediction…</div>
  }

  const correct = prediction.predicted_subtype === prediction.true_subtype

  return (
    <>
      <div className="hero">
        <div className="hero-top">
          <span className="hero-id">{prediction.patient_id}</span>
          <span className="hero-true">true subtype: {prediction.true_subtype}</span>
        </div>
        <div className="hero-predicted">
          <span className="hero-subtype">{prediction.predicted_subtype}</span>
          <span className="hero-confidence">{(prediction.confidence * 100).toFixed(1)}% confidence</span>
          <span className={`hero-match ${correct ? 'correct' : 'incorrect'}`}>
            {correct ? 'matches true subtype' : 'differs from true subtype'}
          </span>
        </div>

        <div className="prob-bars">
          {SUBTYPE_ORDER.map(s => (
            <div className="prob-row" key={s}>
              <span className="prob-label">{s}</span>
              <div className="prob-track">
                <div className="prob-fill" style={{ width: `${prediction.all_probabilities[s] * 100}%` }} />
              </div>
              <span className="prob-value">{(prediction.all_probabilities[s] * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Modality contribution</div>
        {Object.entries(prediction.modality_contribution).map(([mod, weight]) => (
          <div className="modality-row" key={mod}>
            <span className="modality-label">{mod}</span>
            <div className="modality-track">
              <div className="modality-fill" style={{ width: `${weight * 100}%`, background: LAYER_COLORS[mod] }} />
            </div>
            <span className="modality-value" style={{ color: LAYER_COLORS[mod] }}>{(weight * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>

      {/* Embedded Real Patient Similarity Graph Explorer */}
      <PatientGraphExplorer patientId={patientId} onSelectPatient={onSelectPatient} />

      <div className="card">
        <div className="card-title">Top driving genes</div>
        <div className="layer-toggle">
          {['rna', 'mirna', 'meth'].map(l => (
            <button key={l} className={`layer-btn ${layer === l ? 'active' : ''}`} onClick={() => setLayer(l)}>
              {l}
            </button>
          ))}
        </div>
        {loadingGenes && <div className="loading">Computing attribution…</div>}
        {!loadingGenes && genes && (
          <div className="gene-list">
            {genes.top_genes.map(g => (
              <div className="gene-row" key={g.gene}>
                <span className="gene-name">{g.gene}</span>
                <span className={`gene-score ${g.score > 0 ? 'positive' : 'negative'}`}>
                  {g.score > 0 ? '+' : ''}{g.score.toFixed(3)}
                </span>
                <span className="gene-direction">{g.score > 0 ? 'pushes toward' : 'pushes away'}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
