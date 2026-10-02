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

      <div className="card" style={{ border: '1px solid #8247E5', background: 'rgba(130, 71, 229, 0.04)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div className="card-title" style={{ margin: 0, color: '#8247E5', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>⬡ Polygon Blockchain Patient Vault & Immutability</span>
          </div>
          {security?.verification?.is_unaltered && !security?.is_tampered ? (
            <span style={{ fontSize: 11, background: '#10B981', color: '#fff', padding: '2px 8px', borderRadius: 12, fontWeight: 'bold' }}>
              ✓ VERIFIED ON POLYGON
            </span>
          ) : security?.is_tampered ? (
            <span style={{ fontSize: 11, background: '#EF4444', color: '#fff', padding: '2px 8px', borderRadius: 12, fontWeight: 'bold', animation: 'pulse 1.5s infinite' }}>
              ⚠ TAMPER DETECTED
            </span>
          ) : (
            <span style={{ fontSize: 11, background: '#F59E0B', color: '#fff', padding: '2px 8px', borderRadius: 12, fontWeight: 'bold' }}>
              NOT YET ANCHORED
            </span>
          )}
        </div>

        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: '0 0 12px', lineHeight: 1.5 }}>
          Patient records are encrypted off-chain with AES-256-GCM. A salted SHA-256 fingerprint is committed to Polygon Amoy Testnet.
          Any local mutation to gene expression or PAM50 subtype immediately fails verification.
        </p>

        {security && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12, marginBottom: 14 }}>
            <div style={{ background: 'var(--panel)', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase' }}>Pseudonymous Salted Hash (HIPAA de-identified)</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text)', wordBreak: 'break-all' }}>{security.patient_hash}</div>
            </div>
            <div style={{ background: 'var(--panel)', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--line)' }}>
              <div style={{ fontSize: 10, color: 'var(--text-faint)', textTransform: 'uppercase' }}>Data Fingerprint (SHA-256 Digest)</div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: security.is_tampered ? '#EF4444' : 'var(--text)', wordBreak: 'break-all' }}>
                {security.current_fingerprint}
              </div>
            </div>
            {security.verification?.is_anchored && (
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, color: 'var(--text-faint)' }}>
                <span>Network: <strong>Polygon Amoy (80002)</strong> | Block #{security.verification.block_number}</span>
                <a
                  href={security.verification.polygonscan_url}
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: '#8247E5', textDecoration: 'none', fontWeight: 'bold' }}
                >
                  PolygonScan ↗
                </a>
              </div>
            )}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            className="layer-btn"
            style={{ background: '#8247E5', color: '#fff', border: 'none' }}
            disabled={secLoading}
            onClick={() => {
              setSecLoading(true)
              api.anchorPatient(patientId)
                .then(setSecurity)
                .finally(() => setSecLoading(false))
            }}
          >
            {secLoading ? 'Anchoring…' : 'Anchor to Polygon'}
          </button>

          {security?.verification?.is_anchored && !security?.is_tampered && (
            <button
              className="layer-btn"
              style={{ background: '#EF4444', color: '#fff', border: 'none' }}
              disabled={secLoading}
              onClick={() => {
                setSecLoading(true)
                api.simulateTamper(patientId)
                  .then(() => api.securityStatus(patientId).then(setSecurity))
                  .finally(() => setSecLoading(false))
              }}
            >
              Simulate Malicious Tamper
            </button>
          )}

          {security?.is_tampered && (
            <button
              className="layer-btn"
              style={{ background: '#10B981', color: '#fff', border: 'none' }}
              disabled={secLoading}
              onClick={() => {
                setSecLoading(true)
                api.restorePatient(patientId)
                  .then(() => api.securityStatus(patientId).then(setSecurity))
                  .finally(() => setSecLoading(false))
              }}
            >
              Restore Authentic Data
            </button>
          )}

          <button
            className="layer-btn"
            disabled={secLoading}
            onClick={() => {
              setSecLoading(true)
              api.securityStatus(patientId)
                .then(setSecurity)
                .finally(() => setSecLoading(false))
            }}
          >
            Verify Live
          </button>

          <button
            className="layer-btn"
            style={{ background: '#4F46E5', color: '#fff', border: 'none' }}
            disabled={secLoading}
            onClick={() => {
              setSecLoading(true)
              api.anchorCohort()
                .then(() => api.securityStatus(patientId).then(setSecurity))
                .finally(() => setSecLoading(false))
            }}
          >
            Batch Anchor Cohort (646)
          </button>
        </div>

      </div>
    </>
  )
}

