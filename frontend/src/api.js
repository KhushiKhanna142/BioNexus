const BASE = 'http://localhost:8000/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const api = {
  patients: () => get('/patients'),
  predict: (id) => get(`/predict/${id}`),
  explain: (id, layer = 'rna', topK = 10) => get(`/explain/${id}?layer=${layer}&top_k=${topK}`),
  metrics: () => get('/metrics'),
  graphSummary: () => get('/graphs/summary'),
  patientEgoGraph: (layer, patientId, hops = 1) => get(`/graphs/${layer}/patient/${patientId}?hops=${hops}`),
  patientGraphCompare: (patientId) => get(`/graphs/patient/${patientId}/compare`),
  graphOverview: (layer, maxNodes = 250) => get(`/graphs/${layer}/overview?max_nodes=${maxNodes}`),
  profile: (patientId, layer = 'rna', topN = 30) => get(`/profile/${patientId}?layer=${layer}&top_n=${topN}`),
  securityStatus: (id) => get(`/security/patient/${id}`),
  anchorPatient: (id) => fetch(`${BASE}/security/anchor/${id}`, { method: 'POST' }).then(r => r.json()),
  simulateTamper: (id) => fetch(`${BASE}/security/simulate-tamper/${id}`, { method: 'POST' }).then(r => r.json()),
  restorePatient: (id) => fetch(`${BASE}/security/restore/${id}`, { method: 'POST' }).then(r => r.json()),
  cohortStatus: () => get('/security/cohort/status'),
  anchorCohort: () => fetch(`${BASE}/security/cohort/anchor`, { method: 'POST' }).then(r => r.json()),
}


