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
}
