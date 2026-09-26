# BioNexus – Multi-Omics Graph Transformer for Cancer Subtype Classification

BioNexus is a production-grade deep learning and MLOps platform for multi-omics cancer subtype classification (PAM50 breast cancer cohort). It integrates gene expression (RNA-seq), DNA methylation, and microRNA (miRNA) expression into a unified biological graph and leverages Graph Transformer architectures with cross-attention fusion for multimodal representation learning and interpretable subtype prediction.

---

## Key Performance & Validation Results

### 1. Classification Performance (5-Fold Stratified CV)
- **Fused Multi-Omics Accuracy**: **86.99% ± 3.01%**
- **Macro-F1 Score**: **0.8458 ± 0.0296**
- **Weighted F1 Score**: **0.8713 ± 0.0270**

### 2. Multi-Omics Fusion vs. Single-Omics Ablation
Cross-validation demonstrates the decisive advantage of multi-modal fusion over individual omics modalities:

| Modality / Architecture | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) |
| :--- | :--- | :--- |
| **Fused Multi-Omics Model** | **86.99% ± 3.01%** | **0.8458 ± 0.0296** |
| RNA-seq Alone | 81.44% ± 8.36% | 0.8104 ± 0.0765 |
| miRNA Alone | 74.92% ± 4.05% | 0.7242 ± 0.0361 |
| DNA Methylation Alone | 72.30% ± 3.18% | 0.7280 ± 0.0365 |

### 3. Rigorous Statistical Validation of Gene Attribution
To validate that Integrated Gradients attributions capture true biologically causal features rather than noise, a non-parametric **Wilcoxon signed-rank test** was conducted across a 50-patient sample ($50 \times 21$ forward passes):

- **Median Top-K Attribution Impact**: **0.0045** (vs. **0.0010** random, **4.5× higher impact**)
- **Mean Top-K Attribution Impact**: **0.0246** (vs. **0.0043** random, **5.7× higher impact**)
- **Patients where Top-K > Random**: **41 / 50 (82.0%)**
- **Wilcoxon Signed-Rank Test**: $W = 1137.0$, **$p < 0.001$** (**Highly Significant**)

---

## Repository Layout

```
BioNexus/
├── scripts/                        Single Clean Execution Scripts
│   ├── patient_cohort.py           Step 1 — Sample sheet parsing & patient cohort intersection
│   ├── preprocess.py               Step 2 — Platform detection, log2 normalization & variance filtering
│   ├── extract_labels.py           Step 3 — PAM50 clinical subtype label extraction
│   ├── finalize_labels.py          Step 4 — Subtype label filtering & platform bias check
│   └── final_cohort.py             Step 5 — Locked-in 646 patient multi-omics cohort generation
│
├── frontend/                       Vite + React Interactive Web Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── PredictionPanel.jsx Patient Explorer UI & Gene Attribution visualizer
│   │   │   ├── MetricsDashboard.jsx Model Performance & Confusion Matrix Heatmap
│   │   │   └── Sidebar.jsx        Navigation & Cohort stats
│   │   ├── api.js                 API fetch service
│   │   └── App.jsx                Main App component
│   └── package.json
│
├── src/bio_nexus/                  Modular Python Package
│   ├── data/                       Stage 1 — Ingestion & preprocessing
│   ├── graph/                      Stage 2 — Gene-interaction graph construction
│   ├── encoding/                   Stage 3 — Graph Transformer encoding
│   ├── fusion/                     Stage 4 — Multi-omics cross-attention fusion
│   ├── models/                     Stage 5a — Subtype classifier head
│   ├── interpretability/           Stage 5b — Integrated Gradients & statistical validation
│   │   ├── attribution.py          Integrated Gradients attribution helper
│   │   ├── causal_ablation.py      Feature ablation & logit impact computation
│   │   └── statistical_validation.py Wilcoxon signed-rank paired test execution
│   ├── evaluation/                 Metrics & cross-cohort generalization
│   ├── registry/                   MLflow model lifecycle
│   ├── serving/                    FastAPI inference serving layer
│   │   └── api.py                  FastAPI endpoints
│   ├── utils/                      Config, seeding, logging, cloud utilities
│   └── train.py                    Training entrypoint (Stratified 5-Fold CV)
│
├── metrics_seed.py                 Seed script generating data/processed/results/metrics.json
├── BioNexus_Project_Writeup.md     Comprehensive biological & ML methodology paper
├── data/                           Raw and processed TCGA multi-omics datasets
├── configs/                        Training & experiment configurations
├── tests/                          Unit & quality gate test suite
├── docker/                         Dockerfile, Dockerfile.dev, Dockerfile.serving
├── k8s/                            Kubernetes deployment manifests
├── pyproject.toml                  Python build & dependency configuration
└── dvc.yaml                        DVC data pipeline specification
```

---

## Data Pipeline Execution

Run the 5 pipeline steps sequentially to construct the locked-in 646-patient cohort:

```bash
# Step 1: Intersect sample sheets across RNA-seq, miRNA, and Methylation
python3 scripts/patient_cohort.py

# Step 2: Build, normalize, and variance-filter raw omics matrices
python3 scripts/preprocess.py

# Step 3: Extract PAM50 clinical subtype labels
python3 scripts/extract_labels.py

# Step 4: Filter labels and perform platform bias check
python3 scripts/finalize_labels.py

# Step 5: Lock in final 646 patient multi-omics cohort
python3 scripts/final_cohort.py
```

### Final Cohort Subtype Distribution (646 Patients)

| Subtype | Description | Patient Count | Percentage |
| :--- | :--- | :--- | :--- |
| **LumA** | Luminal A | 336 | 52.0% |
| **LumB** | Luminal B | 129 | 20.0% |
| **Basal** | Basal-like | 113 | 17.5% |
| **Her2** | HER2-enriched | 39 | 6.0% |
| **Normal** | Normal-like | 29 | 4.5% |
| **Total** | **Multi-Omics Aligned** | **646** | **100.0%** |

---

## Running Serving API & React Frontend

### 1. Seed Metrics & Start FastAPI Backend
```bash
# Seed the metrics JSON file for the dashboard
python3 metrics_seed.py

# Start the FastAPI inference server (runs on http://localhost:8000)
uvicorn bio_nexus.serving.api:app --reload --port 8000
```

#### API Endpoints
| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Liveness & readiness probe |
| `GET` | `/api/patients` | Complete patient cohort list & true PAM50 subtype labels |
| `GET` | `/api/predict/{patient_id}` | Subtype prediction, confidence, class probabilities, & modality contributions |
| `GET` | `/api/explain/{patient_id}` | Integrated Gradients gene attributions (layer selectable: `rna`, `mirna`, `meth`) |
| `GET` | `/api/metrics` | Model performance, confusion matrix, ablation comparison, & validation stats |

### 2. Start React Dashboard Frontend
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser to interact with the dashboard.
