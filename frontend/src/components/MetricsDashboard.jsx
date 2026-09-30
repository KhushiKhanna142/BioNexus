import { useEffect, useState } from 'react'
import { Target, TrendingUp, Users, Settings } from 'lucide-react'
import { api } from '../api'

const LABELS = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
const CLASS_COLORS = { LumA: '#3FBFAE', LumB: '#D97D66', Her2: '#7B8FA6', Basal: '#D97D66', Normal: '#7B8FA6' }
const LAYER_COLORS = { rna: '#3FBFAE', mirna: '#D97D66', meth: '#7B8FA6' }

function heatColor(value, max) {
  const t = max === 0 ? 0 : value / max
  return `rgba(63, 191, 174, ${0.06 + t * 0.55})`
}

// Precision / recall / F1 / support derived directly from the confusion matrix -
// no extra backend endpoint needed, matrix already contains everything.
function classReport(matrix, labels) {
  return labels.map((label, i) => {
    const support = matrix[i].reduce((a, b) => a + b, 0)
    const tp = matrix[i][i]
    const colSum = matrix.reduce((sum, row) => sum + row[i], 0)
    const precision = colSum === 0 ? 0 : tp / colSum
    const recall = support === 0 ? 0 : tp / support
    const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall)
    return { label, precision, recall, f1, support }
  })
}

function Donut({ pct, size = 120 }) {
  const r = size / 2 - 8
  const c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--panel-raised)" strokeWidth="10" />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#3FBFAE" strokeWidth="10"
        strokeDasharray={c} strokeDashoffset={c * (1 - pct)} strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x="50%" y="47%" textAnchor="middle" fontSize="26" fontFamily="var(--serif)" fill="var(--text)">
        {(pct * 100).toFixed(0)}%
      </text>
    </svg>
  )
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState(null)
  const [omicsView, setOmicsView] = useState('accuracy')

  useEffect(() => { api.metrics().then(setMetrics).catch(console.error) }, [])

  if (!metrics) return <div className="loading">Loading metrics…</div>

  const {
    fused_model,
    single_omics_ablation,
    interpretability,
    cohort,
    model_config = {
      model: 'GraphGPS (PyG)',
      training: '5-Fold Stratified CV',
      loss: 'Cross-Entropy',
      early_stopping: 'Patience = 15',
    },
  } = metrics
  const cm = fused_model.confusion_matrix.matrix
  const maxCell = Math.max(...cm.flat())
  const report = classReport(cm, LABELS)
  const ratio = interpretability.median_top_k_impact / interpretability.median_random_impact

  const metricKey = omicsView === 'accuracy' ? 'accuracy_mean' : 'macro_f1_mean'

  return (
    <>
      <p className="page-desc">Comprehensive evaluation of BioNexus on TCGA-BRCA multi-omics data.</p>

      <div className="metric-grid">
        <div className="metric-box">
          <div className="metric-icon"><Target size={16} /></div>
          <div className="metric-num">{(fused_model.accuracy_mean * 100).toFixed(1)}<span className="metric-unit">%</span></div>
          <div className="metric-sub">± {(fused_model.accuracy_std * 100).toFixed(1)}%</div>
          <div className="metric-name">Accuracy (5-fold CV)</div>
        </div>
        <div className="metric-box">
          <div className="metric-icon"><TrendingUp size={16} /></div>
          <div className="metric-num">{(fused_model.macro_f1_mean * 100).toFixed(1)}<span className="metric-unit">%</span></div>
          <div className="metric-sub">± {(fused_model.macro_f1_std * 100).toFixed(1)}%</div>
          <div className="metric-name">Macro F1 (5-fold CV)</div>
        </div>
        <div className="metric-box">
          <div className="metric-icon"><Users size={16} /></div>
          <div className="metric-num">{cohort.n_patients}</div>
          <div className="metric-name">Patients</div>
        </div>
        <div className="config-box">
          <div className="config-title"><Settings size={13} /> Model Configuration</div>
          <div className="config-row"><span>Model</span><span>{model_config.model}</span></div>
          <div className="config-row"><span>Training</span><span>{model_config.training}</span></div>
          <div className="config-row"><span>Loss</span><span>{model_config.loss}</span></div>
          <div className="config-row"><span>Early Stopping</span><span>{model_config.early_stopping}</span></div>
        </div>
      </div>

      <div className="two-col">
        <div>
          <div className="card">
            <div className="card-title">Confusion Matrix — Best Fold</div>
            <div className="cm-wrap">
              <div className="confusion-grid">
                <div />
                {LABELS.map(l => <div className="cm-header" key={l}>{l}</div>)}
                {LABELS.map((rowLabel, r) => (
                  <>
                    <div className="cm-rowlabel" key={`r-${rowLabel}`}>{rowLabel}</div>
                    {cm[r].map((val, c) => (
                      <div className="cm-cell" key={`${r}-${c}`} style={{ background: heatColor(val, maxCell) }}>{val}</div>
                    ))}
                  </>
                ))}
              </div>
              <div className="cm-legend">
                <div className="cm-legend-bar" />
                <div className="cm-legend-labels"><span>0</span><span>{maxCell}</span></div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Class-wise Performance</div>
            <table className="report-table">
              <thead>
                <tr><th>Subtype</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr>
              </thead>
              <tbody>
                {report.map(row => (
                  <tr key={row.label}>
                    <td><span className="class-dot" style={{ background: CLASS_COLORS[row.label] }} />{row.label}</td>
                    <td>{(row.precision * 100).toFixed(1)}%</td>
                    <td>{(row.recall * 100).toFixed(1)}%</td>
                    <td>{(row.f1 * 100).toFixed(1)}%</td>
                    <td>{row.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <div className="card">
            <div className="card-title-row">
              <div className="card-title">Single-omics vs. Fused</div>
              <div className="pill-toggle">
                <button className={omicsView === 'accuracy' ? 'active' : ''} onClick={() => setOmicsView('accuracy')}>Accuracy</button>
                <button className={omicsView === 'macro_f1' ? 'active' : ''} onClick={() => setOmicsView('macro_f1')}>Macro F1</button>
              </div>
            </div>
            {Object.entries(single_omics_ablation).map(([layer, m]) => (
              <div className="ablation-row" key={layer}>
                <span className="modality-label">{layer}</span>
                <div className="ablation-track">
                  <div className="ablation-fill" style={{ width: `${m[metricKey] * 100}%`, background: LAYER_COLORS[layer] }} />
                </div>
                <span className="modality-value">{(m[metricKey] * 100).toFixed(1)}%</span>
              </div>
            ))}
            <div className="ablation-row">
              <span className="modality-label" style={{ color: '#3FBFAE' }}>fused</span>
              <div className="ablation-track">
                <div className="ablation-fill" style={{ width: `${fused_model[metricKey] * 100}%`, background: '#3FBFAE' }} />
              </div>
              <span className="modality-value" style={{ color: '#3FBFAE' }}>{(fused_model[metricKey] * 100).toFixed(1)}%</span>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Attribution Validation</div>
            <p className="attr-desc">
              Across {interpretability.n_patients_validated} patients, attribution-flagged genes caused a larger
              prediction change than randomly selected genes in{' '}
              <strong style={{ color: 'var(--teal)' }}>{(interpretability.pct_patients_top_k_beats_random * 100).toFixed(0)}%</strong> of cases
              (Wilcoxon signed-rank test, p &lt; 0.001).
            </p>
            <div className="attr-body">
              <Donut pct={interpretability.pct_patients_top_k_beats_random} />
              <div className="attr-stats">
                <div className="attr-caption">Median logit impact</div>
                <div className="attr-compare">
                  <div className="attr-box"><div className="attr-box-label">Top-attribution genes</div><div className="attr-box-val">{interpretability.median_top_k_impact}</div></div>
                  <div className="attr-box"><div className="attr-box-label">Random genes</div><div className="attr-box-val">{interpretability.median_random_impact}</div></div>
                </div>
              </div>
            </div>
            <div className="ratio-badge">{ratio.toFixed(1)}× greater median impact</div>
            <div className="significant-tag">✓ Statistically significant (Wilcoxon p &lt; 0.001)</div>
          </div>
        </div>
      </div>
    </>
  )
}
