import React, { useEffect, useState, useRef, useMemo } from 'react';
import {
  Share2, Search, Bell, ZoomIn, ZoomOut, Target,
  Dna, Check, AlertTriangle, ChevronDown, Info, Layers, RefreshCw,
} from 'lucide-react';
import { api } from '@/api/client';
import {
  SUBTYPE_COLORS,
  type PAM50Subtype,
  type PatientEgoGraphResponse,
  type PatientGraphCompareResponse,
  type Patient,
} from '@/lib/types';

interface PatientGraphExplorerProps {
  initialPatientId?: string;
  patientId?: string;
  onSelectPatient?: (patientId: string) => void;
  className?: string;
}

const LAYER_CONFIG = {
  rna: { label: 'RNA-seq Graph', short: 'RNA-seq', color: '#4F6FE8', border: 'border-blue-500' },
  mirna: { label: 'miRNA Graph', short: 'miRNA', color: '#F47C9C', border: 'border-pink-500' },
  meth: { label: 'Methylation Graph', short: 'Methylation', color: '#34C38F', border: 'border-emerald-500' },
  compare: { label: 'Multi-Omics Comparison', short: 'Compare', color: '#8B6FE0', border: 'border-purple-500' },
};

const PALETTE: Record<string, string> = {
  LumA: '#4F6FE8',
  LumB: '#F47C9C',
  Her2: '#F5B942',
  Basal: '#9C56F6',
  Normal: '#34C38F',
};

export function PatientGraphExplorer({
  initialPatientId,
  patientId,
  onSelectPatient,
  className = '',
}: PatientGraphExplorerProps) {
  const defaultPid = patientId || initialPatientId || 'TCGA-3C-AAAU';
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>(defaultPid);
  const [activeLayer, setActiveLayer] = useState<'rna' | 'mirna' | 'meth' | 'compare'>('rna');
  const [nodeSizeMode, setNodeSizeMode] = useState<'uniform' | 'similarity'>('uniform');
  const [highlightConsensus, setHighlightConsensus] = useState(true);

  // Search & Filter
  const [searchQuery, setSearchQuery] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Data states
  const [egoData, setEgoData] = useState<PatientEgoGraphResponse | null>(null);
  const [compareData, setCompareData] = useState<PatientGraphCompareResponse | null>(null);
  const [compareEgoGraphs, setCompareEgoGraphs] = useState<{
    rna?: PatientEgoGraphResponse;
    mirna?: PatientEgoGraphResponse;
    meth?: PatientEgoGraphResponse;
  }>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Canvas View Controls
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const svgRef = useRef<SVGSVGElement | null>(null);

  // Load Patient List on mount
  useEffect(() => {
    const controller = new AbortController();
    api.getPatients(controller.signal)
      .then(list => {
        setPatients(list);
        if (list.length > 0 && !list.some(p => p.id === selectedPatientId)) {
          setSelectedPatientId(list[0].id);
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError') {
          console.error(err);
        }
      });
    return () => controller.abort();
  }, []);

  // Sync patient if prop changes
  useEffect(() => {
    const pid = patientId || initialPatientId;
    if (pid && pid !== selectedPatientId) {
      setSelectedPatientId(pid);
    }
  }, [patientId, initialPatientId]);

  // Load Ego Graph Data and Compare Data
  const fetchData = () => {
    if (!selectedPatientId) return;
    setLoading(true);
    setError(null);
    const controller = new AbortController();

    if (activeLayer === 'compare') {
      Promise.all([
        api.getPatientGraphCompare(selectedPatientId, controller.signal),
        api.getPatientEgoGraph('rna', selectedPatientId, 1, controller.signal).catch(() => null),
        api.getPatientEgoGraph('mirna', selectedPatientId, 1, controller.signal).catch(() => null),
        api.getPatientEgoGraph('meth', selectedPatientId, 1, controller.signal).catch(() => null),
      ])
        .then(([comp, rnaEgo, mirnaEgo, methEgo]) => {
          setCompareData(comp);
          setCompareEgoGraphs({
            rna: rnaEgo || undefined,
            mirna: mirnaEgo || undefined,
            meth: methEgo || undefined,
          });
          if (rnaEgo) setEgoData(rnaEgo);
        })
        .catch(err => {
          if (err.name !== 'AbortError') {
            setError(err.message || 'Failed to load multi-omics comparison data');
          }
        })
        .finally(() => setLoading(false));
    } else {
      Promise.all([
        api.getPatientEgoGraph(activeLayer, selectedPatientId, 1, controller.signal),
        api.getPatientGraphCompare(selectedPatientId, controller.signal).catch(() => null),
      ])
        .then(([ego, comp]) => {
          if (!ego || !ego.nodes || ego.nodes.length < 11) {
            setError(`Patient ${selectedPatientId} has fewer than 10 nearest neighbors in the ${activeLayer} layer. Invariant requires k>=10.`);
            setEgoData(null);
          } else {
            setEgoData(ego);
            setCompareData(comp);
          }
        })
        .catch(err => {
          if (err.name !== 'AbortError') {
            setError(err.message || 'Failed to load graph data');
          }
        })
        .finally(() => setLoading(false));
    }

    return () => controller.abort();
  };

  useEffect(() => {
    const cancel = fetchData();
    return () => {
      if (cancel) cancel();
    };
  }, [selectedPatientId, activeLayer]);

  const handleSelectPatient = (pid: string) => {
    setSelectedPatientId(pid);
    setIsDropdownOpen(false);
    if (onSelectPatient) {
      onSelectPatient(pid);
    }
  };

  // Zoom & Pan Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.min(Math.max(prev * zoomFactor, 0.6), 2.5));
  };

  const handleRecenter = () => {
    setPan({ x: 0, y: 0 });
    setZoom(1);
  };

  // Helper to build a radial layout for top 10 neighbors of any ego graph
  const buildEgoLayout = (ego: PatientEgoGraphResponse | null, width = 760, height = 480) => {
    if (!ego) return { center: { x: width / 2, y: height / 2 }, neighbors: [], interEdges: [] };

    const cx = width / 2;
    const cy = height / 2;
    const top10 = ego.top_10_neighbors || [];
    const baseRadius = Math.min(width, height) * 0.36;

    const angles = [
      -Math.PI * 0.45,  // Top
      -Math.PI * 0.18,  // Top right
      Math.PI * 0.05,   // Right
      Math.PI * 0.28,   // Bottom right
      Math.PI * 0.55,   // Bottom
      Math.PI * 0.78,   // Bottom left
      Math.PI * 1.05,   // Left
      Math.PI * 1.28,   // Top left
      -Math.PI * 0.05,  // Mid right
      Math.PI * 0.42,   // Mid bottom right
    ];

    const neighbors = top10.map((nbr, idx) => {
      // Visual-only layout geometry (polar-to-Cartesian projection for ego ring)
      const a = angles[idx % angles.length];
      const r = baseRadius + (idx % 2 === 0 ? 12 : -12);
      return {
        ...nbr,
        x: cx + r * Math.cos(a),
        y: cy + r * Math.sin(a),
        color: PALETTE[nbr.pam50] || '#4F6FE8',
      };
    });

    const nbrMap = new Map(neighbors.map(n => [n.patient_id, n]));
    const interEdges: { x1: number; y1: number; x2: number; y2: number; sim: number }[] = [];

    (ego.edges || []).forEach(e => {
      if (e.source !== ego.patient_id && e.target !== ego.patient_id) {
        const u = nbrMap.get(e.source);
        const v = nbrMap.get(e.target);
        if (u && v) {
          interEdges.push({
            x1: u.x,
            y1: u.y,
            x2: v.x,
            y2: v.y,
            sim: e.cosine_similarity,
          });
        }
      }
    });

    return {
      center: { x: cx, y: cy },
      neighbors,
      interEdges,
    };
  };

  const singleLayout = useMemo(() => {
    return buildEgoLayout(egoData, 760, 480);
  }, [egoData]);

  // Subtype counts & percentages for donut
  const subtypeDistribution = useMemo(() => {
    if (!egoData?.top_10_neighbors) return [];
    const counts: Record<string, number> = {
      LumA: 0,
      LumB: 0,
      Her2: 0,
      Basal: 0,
      Normal: 0,
    };
    egoData.top_10_neighbors.forEach(n => {
      if (counts[n.pam50] !== undefined) {
        counts[n.pam50]++;
      }
    });
    const total = egoData.top_10_neighbors.length || 1;
    return Object.entries(counts).map(([st, count]) => ({
      subtype: st as PAM50Subtype,
      count,
      percent: Math.round((count / total) * 100),
      color: PALETTE[st],
    }));
  }, [egoData]);

  const clinical = egoData?.clinical_details || {
    patient_id: selectedPatientId,
    true_pam50: 'Not available',
    predicted_pam50: 'Not available',
    confidence: undefined,
    age: 'Not available',
    er_status: 'Not available',
    pr_status: 'Not available',
    her2_status: 'Not available',
    tumor_grade: 'Not available',
    lymph_node_status: 'Not available',
    histology: 'Not available',
    cancer_type: 'Breast Invasive Carcinoma (TCGA-BRCA)',
  };

  const graphStats = egoData?.graph_stats_details;

  const filteredPatients = patients.filter(p =>
    p.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (p.subtype && p.subtype.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const consensusSet = useMemo(() => {
    return new Set(compareData?.consensus_neighbors || []);
  }, [compareData]);

  return (
    <div className={`space-y-5 bg-[#F8FAFC] min-h-screen p-6 font-sans text-slate-800 ${className}`}>
      {/* Top Bar Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-[#0F172A] tracking-tight">Graph Explorer</h1>
          <p className="text-[13px] text-slate-500 mt-0.5 max-w-3xl leading-normal">
            Explore patient similarity across multi-omics layers. Each node represents a patient and edges show similarity based on the selected omics data.
          </p>
        </div>

        {/* Search, Notifications, Avatar */}
        <div className="flex items-center gap-3 self-end md:self-auto">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search patient ID (e.g. TCGA-XX-XXXX)"
              className="pl-8.5 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg outline-none focus:border-blue-500 w-64 text-slate-800 shadow-2xs placeholder:text-slate-400"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
            />
          </div>

          <button className="relative p-2 text-slate-500 hover:text-slate-800 bg-white border border-slate-200 rounded-lg shadow-2xs">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-rose-500 rounded-full border border-white" />
          </button>

          <div className="w-8 h-8 rounded-full bg-[#1E293B] text-white flex items-center justify-center font-bold text-xs shadow-xs">
            BN
          </div>
        </div>
      </div>

      {/* Select Patient & Quick Clinical Info Strip */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200/90 shadow-2xs flex flex-wrap items-center justify-between gap-4">
        {/* Patient Combobox */}
        <div className="flex items-center gap-2">
          <div className="relative">
            <label className="text-[11px] font-semibold text-slate-500 block mb-1">Select Patient</label>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center justify-between gap-3 px-3 py-1.5 bg-white border border-slate-300 hover:border-slate-400 rounded-lg text-xs font-bold text-slate-900 font-mono min-w-[170px] shadow-2xs"
              >
                <span>{selectedPatientId}</span>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
              </button>

              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="p-2 bg-white hover:bg-slate-50 border border-slate-300 rounded-lg text-slate-600 shadow-2xs"
              >
                <Search className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Dropdown Menu */}
            {isDropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-64 bg-white border border-slate-200 rounded-xl shadow-xl z-50 max-h-60 overflow-y-auto p-1 text-xs">
                <input
                  type="text"
                  placeholder="Filter 646 cohort..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full p-2 border-b border-slate-100 outline-none text-xs text-slate-800"
                  autoFocus
                />
                {filteredPatients.slice(0, 40).map(p => (
                  <button
                    key={p.id}
                    onClick={() => handleSelectPatient(p.id)}
                    className="w-full text-left px-3 py-2 hover:bg-blue-50 flex items-center justify-between rounded-lg transition"
                  >
                    <span className="font-mono font-semibold text-slate-900">{p.id}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Patient Details Badge */}
        <div className="flex items-center gap-3 bg-slate-50/70 border border-slate-100 px-3.5 py-2 rounded-xl">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold">
            <Dna className="w-4.5 h-4.5" />
          </div>
          <div>
            <div className="font-bold text-xs text-slate-900 font-mono">{selectedPatientId}</div>
            <div className="text-[11px] text-slate-500 font-medium">{clinical.cancer_type || 'Breast Invasive Carcinoma (TCGA-BRCA)'}</div>
          </div>
        </div>

        {/* Predicted Subtype */}
        <div>
          <div className="text-[11px] font-semibold text-slate-500 mb-0.5">Predicted Subtype</div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-[#4F6FE8] flex items-center gap-1">
              <span>♦</span> {clinical.predicted_pam50 || 'Not available'}
            </span>
            {clinical.confidence !== undefined && (
              <span className="px-2 py-0.5 bg-[#EEF2FF] text-[#4F6FE8] font-mono font-bold text-xs rounded-md border border-[#E0E7FF]">
                {clinical.confidence.toFixed(2)}
              </span>
            )}
          </div>
        </div>

        {/* Actual Subtype */}
        <div>
          <div className="text-[11px] font-semibold text-slate-500 mb-0.5">Actual Subtype</div>
          <div className="text-xs font-bold text-[#4F6FE8] flex items-center gap-1">
            <span>♦</span> {clinical.true_pam50 || 'Not available'}
          </div>
        </div>

        {/* Clinical Parameters: Age, ER, PR, HER2 */}
        <div className="flex items-center gap-6 border-l border-slate-200 pl-4">
          <div className="text-center">
            <div className="text-[11px] font-semibold text-slate-500">Age</div>
            <div className="text-xs font-bold text-slate-900">{clinical.age || 'Not available'}</div>
          </div>
          <div className="text-center">
            <div className="text-[11px] font-semibold text-slate-500">ER</div>
            <div className={`text-xs font-bold font-mono ${clinical.er_status === 'Positive' ? 'text-emerald-600' : 'text-slate-700'}`}>
              {clinical.er_status === 'Positive' ? '+' : clinical.er_status === 'Negative' ? '-' : clinical.er_status}
            </div>
          </div>
          <div className="text-center">
            <div className="text-[11px] font-semibold text-slate-500">PR</div>
            <div className={`text-xs font-bold font-mono ${clinical.pr_status === 'Positive' ? 'text-emerald-600' : 'text-slate-700'}`}>
              {clinical.pr_status === 'Positive' ? '+' : clinical.pr_status === 'Negative' ? '-' : clinical.pr_status}
            </div>
          </div>
          <div className="text-center">
            <div className="text-[11px] font-semibold text-slate-500">HER2</div>
            <div className={`text-xs font-bold font-mono ${clinical.her2_status === 'Positive' ? 'text-rose-600' : 'text-slate-700'}`}>
              {clinical.her2_status === 'Positive' ? '+' : clinical.her2_status === 'Negative' ? '-' : clinical.her2_status}
            </div>
          </div>
        </div>
      </div>

      {/* Modality Tabs & View Options */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {(['rna', 'mirna', 'meth', 'compare'] as const).map(tabKey => {
            const cfg = LAYER_CONFIG[tabKey];
            const isActive = activeLayer === tabKey;
            return (
              <button
                key={tabKey}
                onClick={() => setActiveLayer(tabKey)}
                className={`px-4 py-2 text-xs font-bold rounded-lg transition-all shadow-2xs ${
                  isActive
                    ? 'bg-[#132247] text-white shadow-xs'
                    : 'bg-white text-slate-600 hover:text-slate-900 hover:bg-slate-50 border border-slate-200'
                }`}
              >
                {cfg.label}
              </button>
            );
          })}
        </div>

        {activeLayer === 'compare' && (
          <label className="flex items-center gap-2 bg-white px-3 py-1.5 rounded-lg border border-slate-200 text-xs text-slate-700 font-semibold cursor-pointer shadow-2xs">
            <input
              type="checkbox"
              checked={highlightConsensus}
              onChange={e => setHighlightConsensus(e.target.checked)}
              className="accent-indigo-600 w-3.5 h-3.5 rounded"
            />
            <span>Highlight consensus neighbors (&ge; 2 layers)</span>
          </label>
        )}
      </div>

      {/* Error State if any */}
      {error && (
        <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-center justify-between text-rose-800 text-xs">
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="w-4 h-4 text-rose-600" />
            <span>{error}</span>
          </div>
          <button
            onClick={fetchData}
            className="flex items-center gap-1.5 px-3 py-1 bg-rose-600 text-white rounded-lg font-bold hover:bg-rose-700 transition"
          >
            <RefreshCw className="w-3 h-3" /> Retry
          </button>
        </div>
      )}

      {/* Main Content Area */}
      {activeLayer !== 'compare' ? (
        /* ================= SINGLE LAYER VIEW (Matches Reference Image 5) ================= */
        <div className="space-y-5">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
            {/* Left: Interactive Canvas Card */}
            <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200/90 shadow-2xs relative overflow-hidden h-[540px]">
              {/* Floating Top-Left Subtype Legend & Node Size */}
              <div className="absolute top-4 left-4 z-10 bg-white/95 backdrop-blur border border-slate-200/80 rounded-xl p-3.5 shadow-sm text-xs space-y-3 pointer-events-auto">
                <div>
                  <div className="font-bold text-slate-800 text-[11px] mb-2">PAM50 Subtype <span className="font-normal text-slate-400">(Node Color)</span></div>
                  <div className="space-y-1.5">
                    {[
                      { name: 'LumA', color: '#4F6FE8' },
                      { name: 'LumB', color: '#F47C9C' },
                      { name: 'Her2', color: '#F5B942' },
                      { name: 'Basal', color: '#9C56F6' },
                      { name: 'Normal', color: '#34C38F' },
                    ].map(st => (
                      <div key={st.name} className="flex items-center gap-2 text-[11px] text-slate-600 font-medium">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: st.color }} />
                        <span>{st.name}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-100">
                  <div className="font-bold text-slate-800 text-[11px] mb-1.5">Node Size</div>
                  <div className="space-y-1 text-[11px] text-slate-600">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="nodeSize"
                        checked={nodeSizeMode === 'uniform'}
                        onChange={() => setNodeSizeMode('uniform')}
                        className="accent-[#132247]"
                      />
                      <span>Uniform</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="nodeSize"
                        checked={nodeSizeMode === 'similarity'}
                        onChange={() => setNodeSizeMode('similarity')}
                        className="accent-[#132247]"
                      />
                      <span>By Similarity</span>
                    </label>
                  </div>
                </div>
              </div>

              {/* Floating Top-Right Zoom & Recenter Controls */}
              <div className="absolute top-4 right-4 z-10 flex flex-col bg-white border border-slate-200/90 rounded-lg shadow-sm overflow-hidden">
                <button
                  onClick={() => setZoom(z => Math.min(z * 1.15, 2.5))}
                  className="p-2 hover:bg-slate-50 text-slate-600 hover:text-slate-900 border-b border-slate-100"
                  title="Zoom In"
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setZoom(z => Math.max(z * 0.85, 0.6))}
                  className="p-2 hover:bg-slate-50 text-slate-600 hover:text-slate-900 border-b border-slate-100"
                  title="Zoom Out"
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <button
                  onClick={handleRecenter}
                  className="p-2 hover:bg-slate-50 text-slate-600 hover:text-slate-900"
                  title="Recenter"
                >
                  <Target className="w-4 h-4" />
                </button>
              </div>

              {/* SVG Interactive Canvas */}
              <svg
                ref={svgRef}
                className="w-full h-full cursor-grab active:cursor-grabbing select-none bg-[#FCFDFF]"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onWheel={handleWheel}
              >
                <defs>
                  <filter id="center-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#4F6FE8" floodOpacity="0.4" />
                  </filter>
                  <filter id="card-shadow" x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#0F172A" floodOpacity="0.08" />
                  </filter>
                </defs>

                <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                  {/* Real inter-neighbor edges */}
                  {singleLayout.interEdges.map((e, idx) => (
                    <line
                      key={`inter-${idx}`}
                      x1={e.x1}
                      y1={e.y1}
                      x2={e.x2}
                      y2={e.y2}
                      stroke="#CBD5E1"
                      strokeWidth={Math.max(1, (e.sim - 0.5) * 4)}
                      strokeOpacity={0.45}
                    />
                  ))}

                  {/* Primary Spokes / Edges from Center to Neighbors */}
                  {singleLayout.neighbors.map((nbr, idx) => {
                    const strokeWidth = Math.max(1.5, (nbr.similarity - 0.6) * 7);
                    return (
                      <line
                        key={`spoke-${nbr.patient_id}-${idx}`}
                        x1={singleLayout.center.x}
                        y1={singleLayout.center.y}
                        x2={nbr.x}
                        y2={nbr.y}
                        stroke="#94A3B8"
                        strokeWidth={strokeWidth}
                        strokeOpacity={0.7}
                      />
                    );
                  })}

                  {/* Top 10 Neighbor Nodes */}
                  {singleLayout.neighbors.map(nbr => {
                    const nodeRadius = nodeSizeMode === 'similarity' ? 8 + (nbr.similarity * 8) : 11;
                    return (
                      <g
                        key={nbr.patient_id}
                        className="cursor-pointer group"
                        onClick={() => handleSelectPatient(nbr.patient_id)}
                      >
                        {/* Circle Node */}
                        <circle
                          cx={nbr.x}
                          cy={nbr.y}
                          r={nodeRadius}
                          fill={nbr.color}
                          stroke="#FFFFFF"
                          strokeWidth={2.5}
                          className="transition-transform duration-200 group-hover:scale-125"
                        />

                        {/* Neighbor Callout Card */}
                        <g transform={`translate(${nbr.x}, ${nbr.y + (nbr.y > singleLayout.center.y ? 14 : -38)})`}>
                          <rect
                            x="-48"
                            y="0"
                            width="96"
                            height="28"
                            rx="6"
                            fill="#FFFFFF"
                            stroke="#E2E8F0"
                            strokeWidth="1"
                            filter="url(#card-shadow)"
                          />
                          <text
                            x="0"
                            y="11"
                            textAnchor="middle"
                            fontSize="8.5"
                            fontWeight="bold"
                            fill="#0F172A"
                            className="font-mono"
                          >
                            {nbr.patient_id}
                          </text>
                          <text
                            x="0"
                            y="22"
                            textAnchor="middle"
                            fontSize="8"
                            fontWeight="500"
                            fill="#64748B"
                          >
                            {nbr.pam50} | {nbr.similarity.toFixed(2)}
                          </text>
                        </g>
                      </g>
                    );
                  })}

                  {/* Center Focus Node */}
                  <g transform={`translate(${singleLayout.center.x}, ${singleLayout.center.y})`}>
                    <circle
                      r="24"
                      fill="none"
                      stroke="#3B82F6"
                      strokeWidth="2"
                      strokeDasharray="4,2"
                      opacity="0.8"
                    />
                    <circle
                      r="16"
                      fill="#0F172A"
                      stroke="#3B82F6"
                      strokeWidth="3.5"
                      filter="url(#center-glow)"
                    />

                    {/* Center Badge Label */}
                    <g transform="translate(0, 24)">
                      <rect
                        x="-60"
                        y="0"
                        width="120"
                        height="32"
                        rx="6"
                        fill="#0F172A"
                        stroke="#1E293B"
                        filter="url(#card-shadow)"
                      />
                      <text
                        x="0"
                        y="13"
                        textAnchor="middle"
                        fontSize="9.5"
                        fontWeight="bold"
                        fill="#FFFFFF"
                        className="font-mono"
                      >
                        {selectedPatientId}
                      </text>
                      <text
                        x="0"
                        y="25"
                        textAnchor="middle"
                        fontSize="8.5"
                        fontWeight="600"
                        fill="#93C5FD"
                      >
                        {clinical.true_pam50} | Conf: {clinical.confidence !== undefined ? clinical.confidence.toFixed(2) : 'N/A'}
                      </text>
                    </g>
                  </g>
                </g>
              </svg>
            </div>

            {/* Right Column: Selected Patient Details & Graph Statistics */}
            <div className="lg:col-span-4 space-y-4">
              {/* Card 1: Selected Patient Details */}
              <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4">
                <h3 className="font-bold text-slate-900 text-sm mb-3">Selected Patient Details</h3>
                <div className="divide-y divide-slate-100 text-xs">
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Patient ID</span>
                    <span className="font-mono font-bold text-slate-900">{selectedPatientId}</span>
                  </div>
                  <div className="py-1.5 flex justify-between items-center">
                    <span className="text-slate-500 font-medium">PAM50 (Actual)</span>
                    <span className="font-bold text-[#4F6FE8]">{clinical.true_pam50}</span>
                  </div>
                  <div className="py-1.5 flex justify-between items-center">
                    <span className="text-slate-500 font-medium">PAM50 (Predicted)</span>
                    <span className="font-bold text-[#4F6FE8]">{clinical.predicted_pam50 || 'Not available'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Prediction Confidence</span>
                    <span className="font-mono font-bold text-slate-900">
                      {clinical.confidence !== undefined ? clinical.confidence.toFixed(2) : 'Not available'}
                    </span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Age</span>
                    <span className="font-medium text-slate-900">{clinical.age || 'Not available'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">ER Status</span>
                    <span className={`font-bold ${clinical.er_status === 'Positive' ? 'text-emerald-600' : 'text-slate-800'}`}>
                      {clinical.er_status || 'Not available'}
                    </span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">PR Status</span>
                    <span className={`font-bold ${clinical.pr_status === 'Positive' ? 'text-emerald-600' : 'text-slate-800'}`}>
                      {clinical.pr_status || 'Not available'}
                    </span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">HER2 Status</span>
                    <span className={`font-bold ${clinical.her2_status === 'Positive' ? 'text-rose-600' : 'text-slate-800'}`}>
                      {clinical.her2_status || 'Not available'}
                    </span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Tumor Grade</span>
                    <span className="font-medium text-slate-900">{clinical.tumor_grade || 'Not available'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Lymph Node Status</span>
                    <span className="font-medium text-slate-900">{clinical.lymph_node_status || 'Not available'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Histology</span>
                    <span className="font-medium text-slate-900 truncate max-w-[170px]" title={clinical.histology}>
                      {clinical.histology || 'Not available'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Card 2: Graph Statistics */}
              <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4">
                <h3 className="font-bold text-slate-900 text-sm mb-3">
                  Graph Statistics ({LAYER_CONFIG[activeLayer].short})
                </h3>
                <div className="divide-y divide-slate-100 text-xs">
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Total Patients</span>
                    <span className="font-mono font-bold text-slate-900">{graphStats?.total_patients != null ? graphStats.total_patients.toLocaleString() : '—'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Total Edges (k=10)</span>
                    <span className="font-mono font-bold text-slate-900">{graphStats?.total_edges != null ? graphStats.total_edges.toLocaleString() : '—'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Neighbors (k)</span>
                    <span className="font-mono font-bold text-slate-900">{graphStats?.neighbors_k != null ? graphStats.neighbors_k : '—'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Mean Similarity (Top 10)</span>
                    <span className="font-mono font-bold text-slate-900">{graphStats?.mean_similarity != null ? graphStats.mean_similarity.toFixed(2) : '—'}</span>
                  </div>
                  <div className="py-1.5 flex justify-between">
                    <span className="text-slate-500 font-medium">Same Subtype Neighbors</span>
                    <span className="font-mono font-bold text-slate-900">{graphStats?.same_subtype_neighbors || '—'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Section: 3 Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Card 1: Top 10 Neighbors Table */}
            <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-slate-900 text-sm mb-3">
                  Top 10 Neighbors ({LAYER_CONFIG[activeLayer].short})
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-100 pb-1.5 text-left font-semibold text-[11px]">
                        <th className="pb-1.5 pr-2">Rank</th>
                        <th className="pb-1.5 pr-2">Patient ID</th>
                        <th className="pb-1.5 pr-2">PAM50</th>
                        <th className="pb-1.5 text-right">Similarity</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-50">
                      {(egoData?.top_10_neighbors || []).map(nbr => (
                        <tr
                          key={nbr.patient_id}
                          className="hover:bg-slate-50/80 cursor-pointer transition"
                          onClick={() => handleSelectPatient(nbr.patient_id)}
                        >
                          <td className="py-1.5 pr-2 text-slate-500 font-medium">{nbr.rank}</td>
                          <td className="py-1.5 pr-2 font-mono font-medium text-slate-800">{nbr.patient_id}</td>
                          <td className="py-1.5 pr-2">
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ backgroundColor: PALETTE[nbr.pam50] || '#4F6FE8' }}
                              />
                              <span className="text-slate-700">{nbr.pam50}</span>
                            </span>
                          </td>
                          <td className="py-1.5 text-right font-mono font-bold text-slate-900">
                            {nbr.similarity.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Card 2: Neighbor Subtype Distribution Donut */}
            <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-slate-900 text-sm mb-2">Neighbor Subtype Distribution</h3>
                <div className="flex items-center justify-between gap-4 my-2">
                  {/* SVG Donut Chart */}
                  <div className="relative w-36 h-36 flex items-center justify-center shrink-0">
                    <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                      {(() => {
                        let accumulatedPercent = 0;
                        return subtypeDistribution.map(item => {
                          const strokeDasharray = `${item.percent * 2.512} ${251.2 - item.percent * 2.512}`;
                          const strokeDashoffset = -accumulatedPercent * 2.512;
                          accumulatedPercent += item.percent;
                          if (item.percent === 0) return null;
                          return (
                            <circle
                              key={item.subtype}
                              cx="50"
                              cy="50"
                              r="40"
                              fill="transparent"
                              stroke={item.color}
                              strokeWidth="18"
                              strokeDasharray={strokeDasharray}
                              strokeDashoffset={strokeDashoffset}
                            />
                          );
                        });
                      })()}
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center pointer-events-none">
                      <span className="text-lg font-extrabold text-slate-900 leading-none">
                        {egoData?.top_10_neighbors?.length || 10}
                      </span>
                      <span className="text-[10px] text-slate-400 font-medium mt-0.5">Neighbors</span>
                    </div>
                  </div>

                  {/* Subtype Legend Breakdown */}
                  <div className="space-y-1.5 text-xs flex-1">
                    {subtypeDistribution.map(item => (
                      <div key={item.subtype} className="flex items-center justify-between text-[11.5px]">
                        <span className="flex items-center gap-2">
                          <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                          <span className="font-medium text-slate-700">{item.subtype}</span>
                        </span>
                        <span className="font-mono text-slate-600 font-medium">
                          {item.count} ({item.percent}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Biological Interpretation Info Banner */}
              <div className="mt-3 p-2.5 bg-blue-50/70 border border-blue-100 rounded-lg flex items-start gap-2 text-xs text-blue-900 leading-relaxed">
                <Info className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
                <p className="text-[11px]">
                  The selected patient's {LAYER_CONFIG[activeLayer].short} neighborhood is mostly composed of {clinical.true_pam50} subtype patients, indicating strong biological alignment with {clinical.true_pam50} cases.
                </p>
              </div>
            </div>

            {/* Card 3: Cross-Modality Neighbor Similarity */}
            <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4 flex flex-col justify-between">
              <div>
                <h3 className="font-bold text-slate-900 text-sm mb-3">Cross-Modality Neighbor Similarity</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-100 pb-1.5 text-left font-semibold text-[11px]">
                        <th className="pb-1.5 pr-2">Patient ID</th>
                        <th className="pb-1.5 pr-2 text-center">RNA</th>
                        <th className="pb-1.5 pr-2 text-center">miRNA</th>
                        <th className="pb-1.5 pr-2 text-center">Methylation</th>
                        <th className="pb-1.5 text-center">Shared</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-50 font-mono text-[11px]">
                      {(egoData?.cross_modality_neighbors || []).map(item => (
                        <tr
                          key={item.patient_id}
                          className="hover:bg-slate-50/80 cursor-pointer transition"
                          onClick={() => handleSelectPatient(item.patient_id)}
                        >
                          <td className="py-1.5 pr-2 font-medium text-slate-800">{item.patient_id}</td>
                          <td className="py-1.5 pr-2 text-center text-slate-700">{item.rna.toFixed(2)}</td>
                          <td className="py-1.5 pr-2 text-center text-slate-700">{item.mirna.toFixed(2)}</td>
                          <td className="py-1.5 pr-2 text-center text-slate-700">{item.methylation.toFixed(2)}</td>
                          <td className="py-1.5 text-center font-bold">
                            {item.shared ? (
                              <span className="text-emerald-600 font-sans font-bold">✓</span>
                            ) : (
                              <span className="text-slate-300 font-sans">-</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* ================= MULTI-OMICS COMPARISON VIEW (3 Side-by-Side Graphs) ================= */
        <div className="space-y-5">
          {/* Overlap Stats Ribbon */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
              <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">RNA &cap; miRNA Overlap</div>
              <div className="text-xl font-extrabold text-slate-900 mt-1">
                {compareData?.overlap_counts?.rna_mirna ?? 0} <span className="text-xs font-normal text-slate-500">neighbors</span>
              </div>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
              <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">RNA &cap; Meth Overlap</div>
              <div className="text-xl font-extrabold text-slate-900 mt-1">
                {compareData?.overlap_counts?.rna_meth ?? 0} <span className="text-xs font-normal text-slate-500">neighbors</span>
              </div>
            </div>
            <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
              <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider">miRNA &cap; Meth Overlap</div>
              <div className="text-xl font-extrabold text-slate-900 mt-1">
                {compareData?.overlap_counts?.mirna_meth ?? 0} <span className="text-xs font-normal text-slate-500">neighbors</span>
              </div>
            </div>
            <div className="bg-indigo-50/80 p-4 rounded-xl border border-indigo-100 shadow-2xs">
              <div className="text-[11px] font-semibold text-indigo-700 uppercase tracking-wider">3-Layer Consensus</div>
              <div className="text-xl font-extrabold text-indigo-900 mt-1">
                {compareData?.overlap_counts?.all_three ?? 0} <span className="text-xs font-normal text-indigo-600">neighbors</span>
              </div>
            </div>
          </div>

          {/* 3 Side-by-Side Ego Subgraphs */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {(['rna', 'mirna', 'meth'] as const).map(layerKey => {
              const layerCfg = LAYER_CONFIG[layerKey];
              const layerEgo = compareEgoGraphs[layerKey];
              const layerLayout = buildEgoLayout(layerEgo || null, 360, 320);
              const layerStats = compareData?.layers?.[layerKey];

              return (
                <div key={layerKey} className="bg-white rounded-xl border border-slate-200/90 shadow-2xs overflow-hidden flex flex-col">
                  {/* Card Header */}
                  <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: layerCfg.color }} />
                      <h4 className="font-bold text-xs text-slate-900">{layerCfg.label}</h4>
                    </div>
                    {layerStats && (
                      <span className="text-[11px] font-mono font-semibold text-slate-500">
                        Purity: {Math.round(layerStats.same_subtype_fraction * 100)}%
                      </span>
                    )}
                  </div>

                  {/* Compact Canvas */}
                  <div className="h-[300px] relative bg-[#FCFDFF]">
                    <svg className="w-full h-full select-none">
                      <g>
                        {/* Primary Spokes */}
                        {layerLayout.neighbors.map((nbr, idx) => {
                          const isConsensus = highlightConsensus && consensusSet.has(nbr.patient_id);
                          return (
                            <line
                              key={`spoke-${layerKey}-${nbr.patient_id}-${idx}`}
                              x1={layerLayout.center.x}
                              y1={layerLayout.center.y}
                              x2={nbr.x}
                              y2={nbr.y}
                              stroke={isConsensus ? '#8B5CF6' : '#94A3B8'}
                              strokeWidth={isConsensus ? 2.5 : 1.5}
                              strokeOpacity={isConsensus ? 0.9 : 0.6}
                            />
                          );
                        })}

                        {/* Neighbor Nodes */}
                        {layerLayout.neighbors.map(nbr => {
                          const isConsensus = highlightConsensus && consensusSet.has(nbr.patient_id);
                          return (
                            <g
                              key={`nbr-${layerKey}-${nbr.patient_id}`}
                              className="cursor-pointer group"
                              onClick={() => handleSelectPatient(nbr.patient_id)}
                            >
                              <circle
                                cx={nbr.x}
                                cy={nbr.y}
                                r={isConsensus ? 10 : 8}
                                fill={nbr.color}
                                stroke={isConsensus ? '#8B5CF6' : '#FFFFFF'}
                                strokeWidth={isConsensus ? 2.5 : 2}
                              />
                              <text
                                x={nbr.x}
                                y={nbr.y + (nbr.y > layerLayout.center.y ? 14 : -10)}
                                textAnchor="middle"
                                fontSize="7.5"
                                fontWeight="600"
                                fill="#0F172A"
                                className="font-mono"
                              >
                                {nbr.patient_id.replace('TCGA-', '')}
                              </text>
                            </g>
                          );
                        })}

                        {/* Center Node */}
                        <g transform={`translate(${layerLayout.center.x}, ${layerLayout.center.y})`}>
                          <circle r="12" fill="#0F172A" stroke={layerCfg.color} strokeWidth="3" />
                          <text
                            x="0"
                            y="18"
                            textAnchor="middle"
                            fontSize="8"
                            fontWeight="bold"
                            fill="#0F172A"
                            className="font-mono"
                          >
                            {selectedPatientId.replace('TCGA-', '')}
                          </text>
                        </g>
                      </g>
                    </svg>
                  </div>

                  {/* Footer Stats */}
                  <div className="p-3 bg-slate-50/70 border-t border-slate-100 text-[11px] text-slate-600 flex justify-between items-center">
                    <span>Mean Sim: <strong className="text-slate-800 font-mono">{layerStats?.mean_similarity.toFixed(2) ?? 'N/A'}</strong></span>
                    <span>10 kNN Neighbors</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Consensus Neighbors Table */}
          <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-4">
            <h3 className="font-bold text-slate-900 text-sm mb-3">
              Consensus Neighbors (&ge; 2 Modalities) for {selectedPatientId}
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-100 pb-1.5 text-left font-semibold text-[11px]">
                    <th className="pb-1.5 pr-2">Patient ID</th>
                    <th className="pb-1.5 pr-2 text-center">In RNA Graph</th>
                    <th className="pb-1.5 pr-2 text-center">In miRNA Graph</th>
                    <th className="pb-1.5 pr-2 text-center">In Methylation Graph</th>
                    <th className="pb-1.5 text-center">Layers Shared</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50 font-mono text-[11px]">
                  {(compareData?.all_neighbors || []).map(pid => {
                    const inRna = compareData?.layers?.rna?.neighbors?.includes(pid) ?? false;
                    const inMirna = compareData?.layers?.mirna?.neighbors?.includes(pid) ?? false;
                    const inMeth = compareData?.layers?.meth?.neighbors?.includes(pid) ?? false;
                    const count = (inRna ? 1 : 0) + (inMirna ? 1 : 0) + (inMeth ? 1 : 0);

                    if (count < 2) return null;

                    return (
                      <tr
                        key={pid}
                        className="hover:bg-slate-50/80 cursor-pointer transition"
                        onClick={() => handleSelectPatient(pid)}
                      >
                        <td className="py-2 pr-2 font-bold text-slate-900">{pid}</td>
                        <td className="py-2 pr-2 text-center">
                          {inRna ? <span className="text-emerald-600 font-sans font-bold">✓</span> : <span className="text-slate-300 font-sans">-</span>}
                        </td>
                        <td className="py-2 pr-2 text-center">
                          {inMirna ? <span className="text-emerald-600 font-sans font-bold">✓</span> : <span className="text-slate-300 font-sans">-</span>}
                        </td>
                        <td className="py-2 pr-2 text-center">
                          {inMeth ? <span className="text-emerald-600 font-sans font-bold">✓</span> : <span className="text-slate-300 font-sans">-</span>}
                        </td>
                        <td className="py-2 text-center">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${count === 3 ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-700'}`}>
                            {count} / 3 layers
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
