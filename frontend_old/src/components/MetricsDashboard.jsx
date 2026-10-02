import { useEffect, useState } from 'react'
import { api } from '../api'

const LABELS = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
const LAYER_COLORS = { rna: '#3FBFAE', mirna: '#D97D66', meth: '#7B8FA6' }

function cellColor(value, max) {
  const intensity = value / max
  return `rgba(63, 191, 174, ${0.08 + intensity * 0.5})`
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    api.metrics().then(setMetrics).catch(console.error)
  }, [])

  if (!metrics) return <div className="loading">Loading metrics…</div>

  const { fused_model, single_omics_ablation, interpretability, cohort } = metrics
  const cm = fused_model.confusion_matrix.matrix
  const maxCell = Math.max(...cm.flat())

  return (
    <>
      <div className="metric-grid">
        <div className="metric-box">
          <div className="metric-num">{(fused_model.accuracy_mean * 100).toFixed(1)}%</div>
          <div className="metric-name">accuracy (5-fold CV)</div>
        </div>
        <div className="metric-box">
          <div className="metric-num">{(fused_model.macro_f1_mean * 100).toFixed(1)}%</div>
          <div className="metric-name">macro F1</div>
        </div>
        <div className="metric-box">
          <div className="metric-num">{cohort?.n_patients ?? '—'}</div>
          <div className="metric-name">patients, {cohort?.n_omics_layers ?? 3} omics layers</div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Confusion matrix — best fold</div>
        <div className="confusion-grid">
          <div />
          {LABELS.map(l => <div className="cm-header" key={l}>{l}</div>)}
          {LABELS.map((rowLabel, r) => (
            <>
              <div className="cm-rowlabel" key={`r-${rowLabel}`}>{rowLabel}</div>
              {cm[r].map((val, c) => (
                <div className="cm-cell" key={`${r}-${c}`} style={{ background: cellColor(val, maxCell) }}>
                  {val}
                </div>
              ))}
            </>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-title">Single-omics vs. fused</div>
        {Object.entries(single_omics_ablation).map(([layer, m]) => (
          <div className="ablation-row" key={layer}>
            <span className="modality-label">{layer}</span>
            <div className="ablation-track">
              <div className="ablation-fill" style={{ width: `${m.accuracy_mean * 100}%`, background: LAYER_COLORS[layer] }} />
            </div>
            <span className="modality-value">{(m.accuracy_mean * 100).toFixed(1)}%</span>
          </div>
        ))}
        <div className="ablation-row">
          <span className="modality-label" style={{ color: '#3FBFAE' }}>fused</span>
          <div className="ablation-track">
            <div className="ablation-fill" style={{ width: `${fused_model.accuracy_mean * 100}%`, background: '#3FBFAE' }} />
          </div>
          <span className="modality-value" style={{ color: '#3FBFAE' }}>{(fused_model.accuracy_mean * 100).toFixed(1)}%</span>
        </div>
      </div>

      {interpretability && (
        <div className="card">
          <div className="card-title">Attribution validation</div>
          <p style={{ fontSize: 13, color: 'var(--text-muted)', margin: 0 }}>
            Across {interpretability.n_patients_validated} patients, attribution-flagged genes caused a larger
            prediction change than randomly selected genes in{' '}
            <strong style={{ color: 'var(--teal)' }}>
              {(interpretability.pct_patients_top_k_beats_random * 100).toFixed(0)}%
            </strong>{' '}
            of cases (Wilcoxon signed-rank test, p &lt; 0.001).
          </p>
        </div>
      )}
    </>
  )
}
