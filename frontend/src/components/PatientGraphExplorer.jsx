import React, { useEffect, useState, useRef, useMemo } from 'react'
import {
  Share2, Network, Eye, Layers, Filter, ZoomIn, ZoomOut, RotateCcw,
  Info, Sparkles, CheckCircle2, AlertTriangle, ArrowRight, Compass
} from 'lucide-react'
import { api } from '../api'

const SUBTYPE_COLORS = {
  LumA: '#3FBFAE',
  LumB: '#3B82F6',
  Her2: '#F59E0B',
  Basal: '#EF4444',
  Normal: '#10B981',
}

const LAYER_INFO = {
  rna: { name: 'RNA Expression', color: '#3FBFAE', short: 'RNA' },
  mirna: { name: 'microRNA', color: '#D97D66', short: 'miRNA' },
  meth: { name: 'DNA Methylation', color: '#7B8FA6', short: 'Meth' },
}

export default function PatientGraphExplorer({ patientId, onSelectPatient }) {
  const [activeTab, setActiveTab] = useState('rna') // 'rna' | 'mirna' | 'meth' | 'compare' | 'global'
  const [hops, setHops] = useState(1)
  const [minSim, setMinSim] = useState(0.0)
  const [subtypeFilter, setSubtypeFilter] = useState('ALL') // 'ALL' or specific subtype
  
  // Data states
  const [egoData, setEgoData] = useState(null)
  const [compareData, setCompareData] = useState(null)
  const [summaryData, setSummaryData] = useState(null)
  const [overviewData, setOverviewData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // UI interaction states
  const [hoveredNode, setHoveredNode] = useState(null)
  const [hoveredEdge, setHoveredEdge] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [nodePositions, setNodePositions] = useState({})
  
  const svgRef = useRef(null)

  // Fetch summary once
  useEffect(() => {
    api.graphSummary()
      .then(setSummaryData)
      .catch(err => console.warn('Could not load graph summary:', err))
  }, [])

  // Fetch single layer ego graph
  useEffect(() => {
    if (!patientId || activeTab === 'compare' || activeTab === 'global') return
    setLoading(true)
    setError(null)
    api.patientEgoGraph(activeTab, patientId, hops)
      .then(data => {
        setEgoData(data)
        initGraphLayout(data)
      })
      .catch(err => {
        setError(err.message || 'Failed to load ego graph')
        setEgoData(null)
      })
      .finally(() => setLoading(false))
  }, [patientId, activeTab, hops])

  // Fetch compare data
  useEffect(() => {
    if (!patientId || activeTab !== 'compare') return
    setLoading(true)
    setError(null)
    api.patientGraphCompare(patientId)
      .then(setCompareData)
      .catch(err => {
        setError(err.message || 'Failed to load multi-layer comparison')
        setCompareData(null)
      })
      .finally(() => setLoading(false))
  }, [patientId, activeTab])

  // Fetch overview data
  useEffect(() => {
    if (activeTab !== 'global') return
    setLoading(true)
    setError(null)
    api.graphOverview('rna', 180)
      .then(data => {
        setOverviewData(data)
        initGraphLayout(data)
      })
      .catch(err => {
        setError(err.message || 'Failed to load graph overview')
        setOverviewData(null)
      })
      .finally(() => setLoading(false))
  }, [activeTab])

  // Simple, deterministic radial-force layout
  const initGraphLayout = (graph) => {
    if (!graph || !graph.nodes) return
    const width = 640
    const height = 480
    const cx = width / 2
    const cy = height / 2
    const positions = {}

    const centerNode = graph.nodes.find(n => n.hop === 0) || graph.nodes[0]
    if (centerNode) {
      positions[centerNode.patient_id] = { x: cx, y: cy }
    }

    const hop1Nodes = graph.nodes.filter(n => n.hop === 1 || (n.hop !== 0 && n.hop !== 2))
    const hop2Nodes = graph.nodes.filter(n => n.hop === 2)

    // Distribute hop 1 in circle
    const r1 = Math.min(width, height) * 0.32
    hop1Nodes.forEach((node, idx) => {
      const angle = (idx / hop1Nodes.length) * 2 * Math.PI - Math.PI / 2
      positions[node.patient_id] = {
        x: cx + r1 * Math.cos(angle) + (Math.sin(idx * 3) * 15),
        y: cy + r1 * Math.sin(angle) + (Math.cos(idx * 2) * 15),
      }
    })

    // Distribute hop 2 in outer circle
    const r2 = Math.min(width, height) * 0.44
    hop2Nodes.forEach((node, idx) => {
      const angle = (idx / hop2Nodes.length) * 2 * Math.PI
      positions[node.patient_id] = {
        x: cx + r2 * Math.cos(angle),
        y: cy + r2 * Math.sin(angle),
      }
    })

    setNodePositions(positions)
    setPan({ x: 0, y: 0 })
    setZoom(1)
  }

  // Filtered nodes and edges
  const { filteredNodes, filteredEdges } = useMemo(() => {
    const currentGraph = activeTab === 'global' ? overviewData : egoData
    if (!currentGraph || !currentGraph.nodes) return { filteredNodes: [], filteredEdges: [] }

    let nodes = currentGraph.nodes
    if (subtypeFilter !== 'ALL') {
      nodes = nodes.filter(n => n.hop === 0 || n.pam50 === subtypeFilter)
    }

    const validIds = new Set(nodes.map(n => n.patient_id))
    const edges = (currentGraph.edges || []).filter(e => {
      if (!validIds.has(e.source) || !validIds.has(e.target)) return false
      if (e.cosine_similarity < minSim) return false
      return true
    })

    return { filteredNodes: nodes, filteredEdges: edges }
  }, [egoData, overviewData, activeTab, subtypeFilter, minSim])

  // Mouse pan & zoom handlers
  const handleMouseDown = (e) => {
    if (e.button !== 0) return
    setIsDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }

  const handleMouseMove = (e) => {
    if (!isDragging) return
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    })
  }

  const handleMouseUp = () => setIsDragging(false)

  const handleWheel = (e) => {
    e.preventDefault()
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(prev => Math.min(Math.max(prev * zoomFactor, 0.5), 3.0))
  }

  return (
    <div className="card graph-explorer-card">
      <div className="card-header-row">
        <div>
          <div className="card-title flex items-center gap-2">
            <Share2 size={18} className="text-teal-400" />
            <span>Patient Graph Explorer</span>
            <span className="badge-pill bg-slate-800 text-slate-300 text-xs px-2 py-0.5 rounded border border-slate-700">
              Cosine kNN (k=10) & GINE Neighborhood
            </span>
          </div>
          <div className="card-subtitle text-xs text-slate-400 mt-1">
            Real patient-similarity neighborhoods (646 nodes). Ground-truth topology used for GraphGPS local message passing.
          </div>
        </div>

        {/* Layer Tabs */}
        <div className="layer-tabs flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            className={`px-3 py-1 text-xs rounded-md font-medium transition ${activeTab === 'rna' ? 'bg-teal-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            onClick={() => setActiveTab('rna')}
          >
            RNA Graph
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md font-medium transition ${activeTab === 'mirna' ? 'bg-orange-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            onClick={() => setActiveTab('mirna')}
          >
            miRNA Graph
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md font-medium transition ${activeTab === 'meth' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            onClick={() => setActiveTab('meth')}
          >
            Methylation Graph
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md font-medium transition ${activeTab === 'compare' ? 'bg-purple-600 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            onClick={() => setActiveTab('compare')}
          >
            All 3 (Compare)
          </button>
          <button
            className={`px-3 py-1 text-xs rounded-md font-medium transition ${activeTab === 'global' ? 'bg-emerald-700 text-white shadow' : 'text-slate-400 hover:text-white'}`}
            onClick={() => setActiveTab('global')}
          >
            Global Layout
          </button>
        </div>
      </div>

      {/* Disclaimers & Model Insight Banner */}
      <div className="graph-banner mt-3 p-3 bg-slate-900/80 border border-slate-800 rounded-lg flex items-start gap-3">
        <Info size={18} className="text-teal-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-slate-300 leading-relaxed">
          <span className="font-semibold text-teal-300">Ground Truth Architecture:</span> Nodes represent real TCGA-BRCA patients (not genes). Edges represent cosine k-nearest neighbors (k=10, symmetrized). GraphGPS executes local GINE convolution across these exact edges, complemented by global multi-head self-attention across all 646 patient nodes.
        </div>
      </div>

      {/* Global Layer Stats Summary Bar */}
      {summaryData && (
        <div className="grid grid-cols-3 gap-3 mt-3">
          {Object.entries(summaryData).map(([lKey, s]) => {
            const isCurrent = activeTab === lKey
            const info = LAYER_INFO[lKey] || { name: lKey, color: '#3FBFAE' }
            return (
              <div
                key={lKey}
                className={`p-2.5 rounded-lg border text-xs transition ${isCurrent ? 'bg-slate-800/90 border-teal-500/60 shadow-sm' : 'bg-slate-900/60 border-slate-800/80'}`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-semibold" style={{ color: info.color }}>{info.name}</span>
                  <span className="text-[11px] text-slate-400">{s.nodes} nodes • {s.edges.toLocaleString()} edges</span>
                </div>
                <div className="flex justify-between items-center text-slate-300 text-[11px]">
                  <span>Mean degree: <strong className="text-white">{s.mean_degree}</strong></span>
                  <span className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700 text-emerald-400">
                    {(s.neighbor_label_purity.overall * 100).toFixed(1)}% Subtype Purity
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Content Area */}
      {loading && (
        <div className="loading-state py-16 text-center text-slate-400">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-teal-400 mb-2" />
          <div>Computing topological ego graph from cosine kNN matrices…</div>
        </div>
      )}

      {error && !loading && (
        <div className="error-box my-4 p-4 bg-red-950/30 border border-red-800/60 rounded-lg text-red-300 text-xs flex items-center gap-2">
          <AlertTriangle size={16} className="text-red-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* SINGLE LAYER EGO GRAPH OR GLOBAL VIEW */}
      {!loading && !error && activeTab !== 'compare' && (
        <div className="graph-view-container mt-4 grid grid-cols-1 lg:grid-cols-4 gap-4">
          
          {/* Main SVG Graph Canvas */}
          <div className="lg:col-span-3 graph-canvas-wrapper relative bg-slate-950 border border-slate-800 rounded-lg overflow-hidden h-[500px]">
            {/* Toolbar Controls */}
            <div className="canvas-toolbar absolute top-3 left-3 z-10 flex items-center gap-2 bg-slate-900/90 backdrop-blur px-2.5 py-1.5 rounded-md border border-slate-700 shadow">
              <span className="text-xs text-slate-400 mr-1">Hops:</span>
              <button
                className={`px-2 py-0.5 text-xs rounded font-medium ${hops === 1 ? 'bg-teal-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                onClick={() => setHops(1)}
              >
                1-Hop
              </button>
              <button
                className={`px-2 py-0.5 text-xs rounded font-medium ${hops === 2 ? 'bg-teal-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                onClick={() => setHops(2)}
              >
                2-Hop
              </button>
              <div className="h-4 w-px bg-slate-700 mx-1" />
              <button
                className="p-1 text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded"
                title="Zoom In"
                onClick={() => setZoom(z => Math.min(z * 1.2, 3))}
              >
                <ZoomIn size={14} />
              </button>
              <button
                className="p-1 text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded"
                title="Zoom Out"
                onClick={() => setZoom(z => Math.max(z * 0.8, 0.5))}
              >
                <ZoomOut size={14} />
              </button>
              <button
                className="p-1 text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded"
                title="Reset View"
                onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}
              >
                <RotateCcw size={14} />
              </button>
            </div>

            {/* Filter Overlay */}
            <div className="canvas-filters absolute top-3 right-3 z-10 flex items-center gap-2 bg-slate-900/90 backdrop-blur px-2.5 py-1.5 rounded-md border border-slate-700 shadow text-xs">
              <span className="text-slate-400">Subtype:</span>
              <select
                className="bg-slate-800 text-slate-200 text-xs px-2 py-0.5 rounded border border-slate-700 outline-none"
                value={subtypeFilter}
                onChange={e => setSubtypeFilter(e.target.value)}
              >
                <option value="ALL">All Subtypes</option>
                {Object.keys(SUBTYPE_COLORS).map(st => (
                  <option key={st} value={st}>{st}</option>
                ))}
              </select>

              <span className="text-slate-400 ml-2">Min Sim:</span>
              <input
                type="range"
                min="0.0"
                max="0.95"
                step="0.05"
                value={minSim}
                onChange={e => setMinSim(parseFloat(e.target.value))}
                className="w-16 accent-teal-500"
              />
              <span className="text-slate-300 font-mono text-[11px]">{minSim.toFixed(2)}</span>
            </div>

            {/* Truncated notice banner if hops=2 capped */}
            {egoData?.stats?.truncated && (
              <div className="absolute bottom-3 left-3 z-10 bg-amber-950/80 border border-amber-600/50 text-amber-200 text-[11px] px-2.5 py-1 rounded shadow">
                2-hop neighborhood capped at 100 most significant nodes for browser performance.
              </div>
            )}

            {/* SVG Graph Drawing */}
            <svg
              ref={svgRef}
              className="w-full h-full cursor-grab active:cursor-grabbing select-none"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onWheel={handleWheel}
            >
              <defs>
                <filter id="glow-center" x="-50%" y="-50%" width="200%" height="200%">
                  <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#3FBFAE" floodOpacity="0.8" />
                </filter>
                <filter id="glow-selected" x="-50%" y="-50%" width="200%" height="200%">
                  <feDropShadow dx="0" dy="0" stdDeviation="5" floodColor="#F59E0B" floodOpacity="0.9" />
                </filter>
              </defs>

              <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                {/* Edges */}
                {filteredEdges.map((e, idx) => {
                  const sPos = nodePositions[e.source]
                  const tPos = nodePositions[e.target]
                  if (!sPos || !tPos) return null
                  const isHovered = hoveredEdge === e || hoveredNode?.patient_id === e.source || hoveredNode?.patient_id === e.target
                  const isCenterIncident = e.source === patientId || e.target === patientId

                  // Cosine sim mapped to stroke width: 0.8->1.5px, 1.0->4.5px
                  const strokeWidth = Math.max(1.2, (e.cosine_similarity - 0.7) * 8)
                  
                  return (
                    <line
                      key={`${e.source}-${e.target}-${idx}`}
                      x1={sPos.x}
                      y1={sPos.y}
                      x2={tPos.x}
                      y2={tPos.y}
                      stroke={isHovered ? '#3FBFAE' : (isCenterIncident ? '#475569' : '#334155')}
                      strokeWidth={isHovered ? strokeWidth + 2 : strokeWidth}
                      strokeDasharray={e.directed_knn_flag ? 'none' : '3,3'}
                      strokeOpacity={isHovered ? 1.0 : (isCenterIncident ? 0.75 : 0.4)}
                      className="transition-colors cursor-pointer"
                      onMouseEnter={() => setHoveredEdge(e)}
                      onMouseLeave={() => setHoveredEdge(null)}
                    />
                  )
                })}

                {/* Nodes */}
                {filteredNodes.map(node => {
                  const pos = nodePositions[node.patient_id]
                  if (!pos) return null

                  const isCenter = node.patient_id === patientId || node.hop === 0
                  const isSelected = selectedNode?.patient_id === node.patient_id
                  const isHovered = hoveredNode?.patient_id === node.patient_id
                  const color = SUBTYPE_COLORS[node.pam50] || '#94A3B8'
                  const radius = isCenter ? 16 : (node.hop === 2 ? 8 : 11)

                  return (
                    <g
                      key={node.patient_id}
                      transform={`translate(${pos.x}, ${pos.y})`}
                      className="cursor-pointer transition-transform"
                      onMouseEnter={() => setHoveredNode(node)}
                      onMouseLeave={() => setHoveredNode(null)}
                      onClick={() => setSelectedNode(node)}
                    >
                      {/* Halo ring for center or selected */}
                      {isCenter && (
                        <circle
                          r={radius + 6}
                          fill="none"
                          stroke="#3FBFAE"
                          strokeWidth={2}
                          strokeDasharray="4,2"
                          className="animate-spin-slow origin-center"
                        />
                      )}

                      {/* Main Node Circle */}
                      <circle
                        r={radius}
                        fill={color}
                        stroke={isCenter ? '#FFFFFF' : (isSelected ? '#F59E0B' : '#0F172A')}
                        strokeWidth={isCenter ? 3 : (isSelected ? 2.5 : 1.5)}
                        filter={isCenter ? 'url(#glow-center)' : (isSelected ? 'url(#glow-selected)' : undefined)}
                        opacity={node.hop === 2 ? 0.75 : 1.0}
                      />

                      {/* Patient Label */}
                      {(isCenter || isSelected || isHovered || filteredNodes.length <= 20) && (
                        <text
                          y={radius + 12}
                          textAnchor="middle"
                          fill={isCenter ? '#3FBFAE' : '#CBD5E1'}
                          fontSize={isCenter ? 11 : 9}
                          fontWeight={isCenter ? 'bold' : '500'}
                          className="pointer-events-none drop-shadow-md"
                        >
                          {node.patient_id}
                        </text>
                      )}
                    </g>
                  )
                })}
              </g>
            </svg>

            {/* Floating Tooltip for Hovered Node */}
            {hoveredNode && (
              <div className="absolute bottom-3 right-3 z-20 bg-slate-900/95 border border-slate-700 p-3 rounded-lg shadow-xl text-xs max-w-xs pointer-events-none backdrop-blur">
                <div className="font-mono font-bold text-teal-300 text-sm flex items-center justify-between mb-1">
                  <span>{hoveredNode.patient_id}</span>
                  <span
                    className="text-[10px] px-1.5 py-0.5 rounded font-sans font-semibold"
                    style={{ backgroundColor: SUBTYPE_COLORS[hoveredNode.pam50] + '33', color: SUBTYPE_COLORS[hoveredNode.pam50] }}
                  >
                    {hoveredNode.pam50}
                  </span>
                </div>
                <div className="space-y-0.5 text-slate-300 text-[11px]">
                  <div>OOF Predicted: <strong className="text-white">{hoveredNode.predicted_pam50 || 'N/A'}</strong></div>
                  <div>Hop Distance: <strong className="text-white">{hoveredNode.hop === 0 ? 'Center Patient' : `${hoveredNode.hop}-hop`}</strong></div>
                  <div>Layer Degree: <strong className="text-white">{hoveredNode.degree}</strong></div>
                  <div>Directed kNN: <strong className="text-white">{hoveredNode.is_directed_neighbor ? 'Yes (Top-10 Outgoing)' : 'Incoming / Symmetrized'}</strong></div>
                </div>
              </div>
            )}

            {/* Floating Tooltip for Hovered Edge */}
            {hoveredEdge && (
              <div className="absolute bottom-3 right-3 z-20 bg-slate-900/95 border border-slate-700 p-2.5 rounded-lg shadow-xl text-xs max-w-xs pointer-events-none backdrop-blur">
                <div className="font-semibold text-slate-200 text-xs mb-1">Cosine kNN Edge</div>
                <div className="text-[11px] text-slate-300 space-y-0.5">
                  <div>Source: <span className="font-mono text-teal-300">{hoveredEdge.source}</span></div>
                  <div>Target: <span className="font-mono text-teal-300">{hoveredEdge.target}</span></div>
                  <div>Cosine Similarity: <strong className="text-white">{hoveredEdge.cosine_similarity.toFixed(4)}</strong></div>
                  <div>Topology: <strong className="text-emerald-400">{hoveredEdge.directed_knn_flag ? 'Direct Top-10 Selection' : 'Mutual Symmetrized Edge'}</strong></div>
                </div>
              </div>
            )}
          </div>

          {/* Right Sidebar: Neighborhood Statistics & Node Inspector */}
          <div className="space-y-4">
            {/* Ego Stats Card */}
            {egoData?.stats && (
              <div className="p-3 bg-slate-900/90 border border-slate-800 rounded-lg text-xs space-y-2">
                <div className="font-semibold text-slate-200 flex items-center justify-between">
                  <span>Neighborhood Stats</span>
                  <span className="text-teal-400 font-mono text-[11px]">
                    {egoData.stats.n_nodes} nodes • {egoData.stats.n_edges} edges
                  </span>
                </div>

                <div className="p-2 bg-slate-950 rounded border border-slate-800/80">
                  <div className="text-[11px] text-slate-400">Subtype Concordance:</div>
                  <div className="text-base font-bold text-teal-300">
                    {(egoData.stats.same_subtype_fraction * 100).toFixed(1)}% {egoData.center_node.pam50}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Mean neighbor cosine similarity: <strong>{egoData.stats.mean_similarity.toFixed(4)}</strong>
                  </div>
                </div>

                {/* Subtype Breakdown Pills */}
                <div>
                  <div className="text-[11px] text-slate-400 mb-1">Neighbor Subtype Distribution:</div>
                  <div className="space-y-1">
                    {Object.entries(egoData.stats.subtype_counts).map(([st, cnt]) => {
                      const pct = Math.round((cnt / (egoData.stats.n_nodes - 1 || 1)) * 100)
                      return (
                        <div key={st} className="flex items-center justify-between text-[11px]">
                          <span className="flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: SUBTYPE_COLORS[st] }} />
                            <span className="text-slate-300">{st}</span>
                          </span>
                          <span className="text-slate-400 font-mono">{cnt} ({pct}%)</span>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Selected Node Details & Pivot Button */}
            {selectedNode && (
              <div className="p-3 bg-slate-900 border border-teal-500/40 rounded-lg text-xs space-y-2 animate-fadeIn">
                <div className="font-semibold text-teal-300 flex items-center justify-between">
                  <span>Inspected Patient</span>
                  <button
                    className="text-slate-400 hover:text-white"
                    onClick={() => setSelectedNode(null)}
                  >
                    ✕
                  </button>
                </div>
                <div className="font-mono text-sm font-bold text-white">{selectedNode.patient_id}</div>
                <div className="space-y-1 text-slate-300 text-[11px]">
                  <div>True Subtype: <strong style={{ color: SUBTYPE_COLORS[selectedNode.pam50] }}>{selectedNode.pam50}</strong></div>
                  <div>Predicted Subtype: <strong className="text-slate-200">{selectedNode.predicted_pam50 || 'N/A'}</strong></div>
                  <div>Graph Degree: <strong className="text-slate-200">{selectedNode.degree}</strong></div>
                  {selectedNode.fold_id !== undefined && <div>CV Fold: <strong className="text-slate-200">Fold {selectedNode.fold_id}</strong></div>}
                </div>
                {selectedNode.patient_id !== patientId && onSelectPatient && (
                  <button
                    className="w-full mt-2 py-1.5 bg-teal-600 hover:bg-teal-500 text-white font-medium rounded text-xs transition flex items-center justify-center gap-1"
                    onClick={() => {
                      onSelectPatient(selectedNode.patient_id)
                      setSelectedNode(null)
                    }}
                  >
                    <span>Pivot Explorer to this Patient</span>
                    <ArrowRight size={13} />
                  </button>
                )}
              </div>
            )}

            {/* Legend */}
            <div className="p-3 bg-slate-900/60 border border-slate-800 rounded-lg text-[11px] space-y-1.5">
              <div className="font-semibold text-slate-300 mb-1">Graph Legend</div>
              <div className="grid grid-cols-2 gap-1.5">
                {Object.entries(SUBTYPE_COLORS).map(([st, c]) => (
                  <div key={st} className="flex items-center gap-1.5 text-slate-300">
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c }} />
                    <span>{st}</span>
                  </div>
                ))}
              </div>
              <div className="pt-2 border-t border-slate-800 space-y-1 text-slate-400 text-[10px]">
                <div><strong className="text-slate-300">Solid line:</strong> Directed Top-10 kNN</div>
                <div><strong className="text-slate-300">Dashed line:</strong> Incoming Symmetrized Edge</div>
                <div><strong className="text-slate-300">Edge width:</strong> Cosine Similarity (0.8 - 1.0)</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MULTI-LAYER COMPARE VIEW ('All 3' Tab) */}
      {!loading && !error && activeTab === 'compare' && compareData && (
        <div className="compare-view-container mt-4 space-y-4">
          
          {/* Side by Side Layer Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['rna', 'mirna', 'meth'].map(lKey => {
              const info = LAYER_INFO[lKey]
              const comp = compareData.layers[lKey]
              if (!comp) return null
              return (
                <div key={lKey} className="p-3.5 bg-slate-900/90 border border-slate-800 rounded-lg text-xs space-y-2">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="font-bold text-sm" style={{ color: info.color }}>{info.name}</span>
                    <span className="bg-slate-800 px-2 py-0.5 rounded text-slate-300 text-[11px]">
                      {comp.count} neighbors
                    </span>
                  </div>
                  <div className="space-y-1 text-slate-300">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Subtype Concordance:</span>
                      <strong className="text-teal-300">{(comp.same_subtype_fraction * 100).toFixed(1)}% {compareData.true_pam50}</strong>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Mean Cosine Similarity:</span>
                      <strong className="text-white">{comp.mean_similarity.toFixed(4)}</strong>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-slate-800/80">
                    <div className="text-[11px] text-slate-400 mb-1">Top Connected Patients:</div>
                    <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto pr-1">
                      {comp.neighbors.slice(0, 8).map(nid => (
                        <button
                          key={nid}
                          className="font-mono text-[10px] bg-slate-950 hover:bg-teal-950 hover:text-teal-300 text-slate-300 px-1.5 py-0.5 rounded border border-slate-800 transition"
                          onClick={() => onSelectPatient && onSelectPatient(nid)}
                        >
                          {nid}
                        </button>
                      ))}
                      {comp.neighbors.length > 8 && (
                        <span className="text-[10px] text-slate-500 self-center">+{comp.neighbors.length - 8} more</span>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Overlap & Consensus Card */}
          <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="font-semibold text-slate-200 text-sm mb-2 flex items-center gap-2">
                <Sparkles size={16} className="text-purple-400" />
                <span>Multi-Omics Consensus Neighbors</span>
              </div>
              <div className="text-xs text-slate-400 mb-3 leading-relaxed">
                Patients who share high similarity with <strong className="text-white font-mono">{patientId}</strong> across 2 or more distinct omics graphs. These provide high-confidence multi-modal evidence for the cross-attention fusion layer.
              </div>

              {compareData.consensus_neighbors.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {compareData.consensus_neighbors.map(cid => (
                    <span
                      key={cid}
                      className="px-2 py-1 bg-purple-950/60 border border-purple-600/40 text-purple-200 rounded font-mono text-xs cursor-pointer hover:bg-purple-900"
                      onClick={() => onSelectPatient && onSelectPatient(cid)}
                      title="Click to inspect this patient"
                    >
                      {cid} ★
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-slate-500 italic p-2 bg-slate-950 rounded">
                  No multi-layer overlap in 1-hop kNN — indicates each omics layer captures complementary, orthogonal biological signals for this patient.
                </div>
              )}
            </div>

            <div>
              <div className="font-semibold text-slate-200 text-sm mb-2 flex items-center gap-2">
                <Compass size={16} className="text-teal-400" />
                <span>Inter-Layer Neighbor Overlap</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400 text-[11px]">RNA ∩ miRNA</div>
                  <div className="text-lg font-bold text-white font-mono">{compareData.overlap_counts.rna_mirna} patients</div>
                </div>
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400 text-[11px]">RNA ∩ Methylation</div>
                  <div className="text-lg font-bold text-white font-mono">{compareData.overlap_counts.rna_meth} patients</div>
                </div>
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400 text-[11px]">miRNA ∩ Methylation</div>
                  <div className="text-lg font-bold text-white font-mono">{compareData.overlap_counts.mirna_meth} patients</div>
                </div>
                <div className="p-2.5 bg-slate-950 rounded border border-slate-800">
                  <div className="text-slate-400 text-[11px]">All 3 Layers</div>
                  <div className="text-lg font-bold text-purple-300 font-mono">{compareData.overlap_counts.all_three} patients</div>
                </div>
              </div>
            </div>
          </div>

          {/* Biological & GraphGPS Fusion Explanatory Note */}
          <div className="p-3.5 bg-indigo-950/30 border border-indigo-800/40 rounded-lg text-xs text-indigo-200 leading-relaxed">
            <strong className="text-indigo-300">Why this matters for GraphGPS Subtyping:</strong> Each omics modality constructs an independent topological manifold. The local GINE message-passing layers aggregate neighborhood profiles per modality. When neighborhood subtype purity is high across multiple modalities, cross-attention fusion achieves high predictive confidence.
          </div>
        </div>
      )}
    </div>
  )
}
