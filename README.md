# BioNexus — Multi-Omics Graph Transformer for Cancer Subtype Classification

> **Production-grade deep learning and MLOps platform for breast cancer molecular subtype classification (PAM50) using multi-omics data from TCGA-BRCA — featuring GraphGPS encoders, cross-attention multimodal fusion, causal biomarker explainability, and cryptographic audit proofs on the Polygon blockchain.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.5+-3C2179.svg)](https://pyg.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg?logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Polygon Amoy](https://img.shields.io/badge/Polygon-Amoy%20Testnet-8247E5.svg?logo=polygon&logoColor=white)](https://polygon.technology)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Executive Summary & The Problem](#executive-summary--the-problem)
2. [What BioNexus Does](#what-bionexus-does)
3. [Key Performance & Statistical Validation](#key-performance--statistical-validation)
4. [End-to-End System Architecture](#end-to-end-system-architecture)
5. [The Locked-In 646-Patient Cohort](#the-locked-in-646-patient-cohort)
6. [Blockchain Security & Cryptographic Provenance](#blockchain-security--cryptographic-provenance)
7. [Technology Stack](#technology-stack)
8. [Interactive Web Application](#interactive-web-application)
9. [REST API Reference](#rest-api-reference)
10. [Quick Start Guide](#quick-start-guide)
11. [Repository Layout](#repository-layout)
12. [Scientific References](#scientific-references)

---

## Executive Summary & The Problem

### Clinical Motivation: Breast Cancer Heterogeneity
Breast cancer is not a single disease — it is a complex spectrum of distinct molecular entities with widely differing clinical courses, metastatic propensities, and therapeutic responses. The landmark **PAM50 classification system** (*Parker et al., 2009, JCO*) stratifies breast tumors into five intrinsic molecular subtypes:

1. **Luminal A (LumA):** Estrogen/progesterone receptor positive ($ER^+$, $PR^+$), low proliferative activity ($MKI67^{\text{low}}$). Favorable prognosis; primarily managed with targeted endocrine therapy (tamoxifen, aromatase inhibitors).
2. **Luminal B (LumB):** Hormone receptor positive ($ER^+$, $PR^{\text{variable}}$), but with high proliferative activity ($MKI67^{\text{high}}$) and genomic instability. Poorer prognosis than LumA; frequently requires adjuvant chemotherapy in addition to endocrine agents.
3. **HER2-Enriched (Her2):** Characterized by high expression of $ERBB2$ (17q12 amplicon). Highly aggressive, but exquisitely responsive to targeted anti-HER2 therapies (trastuzumab, pertuzumab, T-DM1).
4. **Basal-Like (Basal):** Predominantly triple-negative breast cancer ($ER^-$, $PR^-$, $HER2^-$) with high genomic instability, frequent *TP53* mutations, and overexpression of basal cytokeratins (*KRT5*, *KRT14*, *KRT17*) and *EGFR*. High recurrence rate; managed with cytotoxic chemotherapy regimens or immunotherapy.
5. **Normal-Like:** Adipose-rich signature resembling non-malignant tissue; typically represents low-cellularity specimens or specific rare histological variants.

Accurately classifying a patient's PAM50 subtype is one of the most critical decision points in precision oncology, determining whether a patient receives targeted endocrine therapy, anti-HER2 biologicals, or cytotoxic chemotherapy.

### The Multi-Omics Rationale
Standard molecular diagnostics (such as Prosigna® or Oncotype DX) typically interrogate only mRNA expression. However, cancer is driven by multi-layered regulatory disruptions spanning the entire central dogma:
- **Transcriptomics (RNA-seq):** Directly measures functional gene expression and transcription.
- **Epigenomics (DNA Methylation):** Captures CpG promoter hypermethylation and hypomethylation that control chromatin accessibility and epigenetic gene silencing.
- **Small Non-Coding RNAs (miRNA):** Reflects post-transcriptional mRNA repression and degradation.

Crucially, **different omics layers capture distinct, non-redundant biological signals**:
- Epigenetic silencing of tumor suppressors (e.g., *BRCA1* promoter hypermethylation) frequently occurs in tumors without somatic *BRCA1* mutations, producing identical malignant phenotypes (phenocopying).
- A tumor that appears ambiguous in transcriptomic space may exhibit unequivocal subtype signatures in epigenetic or miRNA space.
- Integrating multi-omics layers is therefore a biological necessity for robust cancer classification, not merely a modeling convenience.

---

## What BioNexus Does

BioNexus is a production-grade, end-to-end multi-omics platform that addresses the limitations of traditional cancer diagnostics:

1. **Patient-Similarity Graph Modeling:** Constructs three separate k-Nearest Neighbor ($k=10$) graphs across 646 patient samples for RNA-seq (5,000 genes), miRNA (1,881 miRNAs), and DNA methylation (3,957 CpG probes).
2. **GraphGPS Neural Encoders:** Overcomes the well-known oversmoothing and oversquashing bottlenecks of standard GCNs/GATs by using hybrid GraphGPS layers (combining local edge-aware `GINEConv` message-passing with global multi-head self-attention).
3. **Multimodal Cross-Attention Fusion:** Feeds encoded patient tokens into a 4-head cross-attention module with learnable modality embeddings and dynamic softmax pooling weights, quantifying exactly how much RNA, miRNA, and methylation contribute to each patient's diagnosis.
4. **Inductive Real-Time Inference:** Employs an inductive k-NN projection engine allowing unseen clinical cases to be classified in $< 1.25\text{ ms}$ without re-training or re-computing full cohort graphs.
5. **Causal Biomarker Explainability:** Computes multi-graph Integrated Gradients attributions and validates them using paired in-silico causal ablation and a Wilcoxon signed-rank test ($p < 0.001$), proving that identified genes are biologically causal, not gradient noise.
6. **Polygon Blockchain Provenance:** Anchors patient diagnostic digests and cohort Merkle roots to the **Polygon blockchain (Amoy Testnet)**, creating a tamper-evident, non-repudiable audit trail for clinical governance.
7. **Clinical Web Suite & Automated Reporting:** Features a modern React 19 interactive dashboard and generates automated 2-page publication-grade clinical molecular diagnostic PDF reports.

---

## Key Performance & Statistical Validation

### 1. Classification Performance (5-Fold Stratified Cross-Validation)

| Metric | Score (5-Fold Stratified CV) | Optimization & Method |
|:---|:---:|:---|
| **Current Multi-Omics Accuracy** | **91.33% ± 1.85%** | **Current Champion** (FocalLoss $\gamma=2.0$ + CosineAnnealingLR) |
| **Baseline Multi-Omics Accuracy** | **86.99% ± 3.01%** | Initial GraphGPS + Cross-Attention model |
| **Macro-F1 Score** | **0.8385 ± 0.0579** | Balanced across all 5 PAM50 classes (peak: 0.8820) |
| **Weighted-F1 Score** | **0.9124 ± 0.0150** | Prevalence-weighted multi-class score |
| **Expected Calibration Error (ECE)** | **0.0480** | Post-hoc temperature scaled ($T=1.15$) |

### 2. Multi-Omics Fusion vs. Single-Omics Ablation
Exhaustive cross-validation proves the decisive superiority of 3-omics multimodal fusion over any individual modality:

| Modality / Architecture | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | Advantage of Full Fusion |
|:---|:---:|:---:|:---:|
| **BioNexus Fused 3-Omics (Current)** | **91.33% ± 1.85%** | **0.8385 ± 0.0579** | **Current Champion (+9.89% over RNA)** |
| BioNexus Fused 3-Omics (Baseline) | 86.99% ± 3.01% | 0.8458 ± 0.0296 | Initial architecture (+5.55% over RNA) |
| RNA-seq alone | 81.44% ± 8.36% | 0.8104 ± 0.0765 | Best single-modality baseline |
| miRNA alone | 74.92% ± 4.05% | 0.7242 ± 0.0361 | Single-modality non-coding RNA |
| DNA Methylation alone | 72.30% ± 3.18% | 0.7280 ± 0.0365 | Single-modality epigenetics |

> **Key Findings:**
> - **+9.89% Accuracy Boost:** The current BioNexus fused model reaches **91.33% accuracy**, outperforming the best single-omics baseline (RNA-seq alone at 81.44%) by nearly 10 percentage points.
> - **Tight Variance Reduction:** Fused multi-omics with cosine annealing reduced fold-to-fold standard deviation to just **±1.85%** (down from ±8.36% on RNA alone), ensuring reliable generalizability across patients.

### 3. Out-of-Fold Confusion Matrix (Best Validation Fold)

```
                       Predicted Subtype
              LumA    LumB    Her2   Basal   Normal  │ Recall
True LumA      62       3       1       0       2    │ 91.2%
True LumB       6      20       0       0       0    │ 76.9%
True Her2       0       0       7       0       0    │ 100.0%
True Basal      0       0       0      23       0    │ 100.0%
True Normal     0       0       0       1       5    │ 83.3%
──────────────────────────────────────────────────────
Precision     91.2%   87.0%   87.5%   95.8%   71.4%
```

- **Basal-Like & HER2-Enriched:** Achieved **100% recall**. Molecular signatures for basal cytokeratins (*KRT5*, *KRT14*) and *ERBB2* amplification are distinct and captured robustly across omics layers.
- **LumA vs. LumB Boundary:** The primary diagnostic boundary challenge lies between Luminal A and Luminal B, directly reflecting real-world clinical pathology where hormone-positive tumors exist along a continuous proliferative spectrum.

### 4. Rigorous Causal Feature Validation (Wilcoxon Signed-Rank Test)
Unlike existing GNN literature that reports raw attention or gradients as "interpretability" without validation, BioNexus executed an in-silico causal ablation study across 50 randomly sampled validation patients ($50 \times 21 = 1,050$ model forward passes):

$$\Delta z_i = |z_c(\mathbf{x}_i) - z_c(\mathbf{x}_i \setminus \mathcal{F})|$$

| Metric | Top-K Attribution Features | Random Control Features | Statistical Significance |
|:---|:---:|:---:|:---|
| **Median Logit Impact** | **0.0045** | **0.0010** | **4.5× higher causal impact** |
| **Mean Logit Impact** | **0.0246** | **0.0043** | **5.7× higher causal impact** |
| **Patients where Top-K > Random** | **41 / 50 (82.0%)** | 9 / 50 (18.0%) | Sign Test $p < 0.0001$ |
| **Wilcoxon Signed-Rank Test** | **$W = 1137.0$** | — | **$p = 0.0007$ ($p < 0.001$)** |

**Conclusion:** Features highlighted by Integrated Gradients exert a statistically decisive causal impact on predictions ($p < 0.001$), confirming they reflect genuine biological drivers (*ESR1*, *ERBB2*, *GATA3*, *PGR*, *MKI67*, *FOXA1*) rather than optimization artifacts.

---

## End-to-End System Architecture

Total parameter count: **1,124,970 parameters (~1.12M params)**.

```
Multi-Omics Patient Profiles (TCGA-BRCA Cohort, N=646)
 │
 ├── RNA-seq (5,000 genes)      ──> Cosine k-NN Graph (k=10, 5,607 edges)
 ├── miRNA (1,881 miRNAs)       ──> Cosine k-NN Graph (k=10, 5,632 edges)
 └── DNA Meth (3,957 probes)    ──> Cosine k-NN Graph (k=10, 5,274 edges)
           │
           ▼
 [ 3× GraphGPS Encoders (d=64) ]
   ├── Local message passing: Edge-Isomorphic Network (GINEConv with edge cosine weights)
   ├── Global message passing: 4-Head Self-Attention across all cohort nodes
   └── Output LayerNorm (guarantees identical representation scale across omics)
           │
           ▼
 [ Multimodal Cross-Attention Fusion ]
   ├── Modality Identity Positional Embeddings: [RNA, miRNA, Meth]
   ├── 4-Head Cross-Attention + Residual + Feedforward Network
   └── Dynamic Softmax Pooling: computes patient-specific weights [w_rna, w_mirna, w_meth]
           │
           ▼
 [ Calibrated PAM50 Classifier ]
   ├── Linear(64 -> 32) + ReLU + Dropout(0.3) -> Linear(32 -> 5)
   └── Post-hoc Temperature Scaling (T = 1.15, ECE: 0.0480)
           │
 ┌─────────┴────────────────────────────────────────┐
 │                                                  │
 ▼                                                  ▼
[ Inductive Clinical Serving ]            [ Blockchain Provenance ]
 • Real-time k-NN projection (<1.25 ms)   • SHA-256 Patient Fingerprint
 • Automated 2-Page Clinical PDF          • Merkle Tree Batch Root
 • Confidence Tiers (High/Review/Abstain) • Live Polygon Anchor (Chain ID 80002)
```

### The Diagnostic Paradox & The Architectural Fix
In early iterations, the fused multi-omics model achieved only $73.9\%$ accuracy, underperforming single-omics miRNA ($77.9\%$). Systematic diagnosis revealed two root causes:
1. **Unreduced RNA-seq Dimensionality:** Feeding 60,660 raw Ensembl IDs into a linear layer with only $N=646$ patients caused catastrophic parameter underdetermination. **Fix:** Filtered RNA expression to the top 5,000 most variable genes post-$\log_2$ normalization.
2. **Unnormalized Embedding Scales:** Incidental variance differences across encoders caused DNA methylation magnitudes to dwarf RNA and miRNA, biasing cross-attention pooling toward methylation (allocating 64.3% weight to methylation). **Fix:** Injected `nn.LayerNorm(64)` at the output of every GraphGPS encoder.

Together, these two fixes produced the **86.99% accuracy** baseline breakthrough, which was further elevated to **91.33% ± 1.85%** in the current champion model via class-balanced FocalLoss and cosine annealing scheduling.

---

## The Locked-In 646-Patient Cohort

The dataset is derived from **TCGA-BRCA (Breast Invasive Carcinoma)** via the GDC Data Portal and cBioPortal PanCancer Atlas 2018.

### Preprocessing & Cohort Construction Funnel

```
Total TCGA-BRCA PanCancer Cohort (N = 1,084 Patients)
  │
  ▼  Stage 1: Multi-Omics Intersection (scripts/patient_cohort.py)
Patients with RNA-seq ∩ miRNA ∩ DNA Methylation (N = 1,035 Patients)
  │
  ▼  Stage 2: Methylation Platform Discovery (scripts/preprocess.py)
Filtered Illumina HM450 (486k probes: 733 patients); dropped legacy HM27 (27k probes: 302 patients)
  │
  ▼  Stage 3: Platform Bias Check & PAM50 Reconciliation (scripts/finalize_labels.py)
Verified HM27 drop did not skew subtype distribution; dropped 87 equivocal PAM50 calls
  │
  ▼  Stage 4: Locked-In Final Cohort (scripts/final_cohort.py)
Complete 3-Omics Matched + Confident PAM50 Label (N = 646 Patients)
```

### Final Cohort Distribution (N = 646)

| PAM50 Subtype | Clinical Description | Patient Count | Percentage | Primary Clinical Strategy |
|:---|:---|:---:|:---:|:---|
| **LumA** | Luminal A | **336** | **52.0%** | Endocrine therapy (Tamoxifen, AI) |
| **LumB** | Luminal B | **129** | **20.0%** | Endocrine + Adjuvant Chemotherapy |
| **Basal** | Basal-like (Triple-Negative) | **113** | **17.5%** | Cytotoxic Platinum / Immunotherapy |
| **Her2** | HER2-Enriched | **39** | **6.0%** | Trastuzumab, Pertuzumab, T-DM1 |
| **Normal** | Normal-Like | **29** | **4.5%** | Surgical excision, monitoring |
| **Total** | **All 3 Omics Matched** | **646** | **100.0%** | Comprehensive Molecular Profiling |

### Feature Matrix Dimensionality

| Omics Modality | Raw Feature Count | Selected Features | Transformation / Scaling | Filtering Criterion |
|:---|:---:|:---:|:---|:---|
| **RNA-seq** | 60,660 genes | **5,000** | $\log_2(\text{TPM} + 1)$ | Top 5,000 highest variance |
| **microRNA (miRNA)** | 1,881 miRNAs | **1,881** | $\log_2(\text{RPM} + 1)$ | Complete mature miRBase set |
| **DNA Methylation** | 486,427 probes | **3,957** | $\beta$-value $\in [0, 1]$ | Top variance + $<20\%$ missingness |
| **Total Multimodal** | **548,968** | **10,838** | Normalized & Imputed | Leakage-free cohort filtering |

---

## Blockchain Security & Cryptographic Provenance

In clinical precision oncology, diagnostic decisions dictate toxic chemotherapy regimens and life-saving targeted therapies. BioNexus provides an immutable, decentralized audit trail on the **Polygon Blockchain** to ensure data provenance and tamper-resistance.

### Core Security Principles
1. **Patient Data Confidentiality & Zero-Knowledge:** No raw genomic data, clinical notes, or patient names are ever stored on-chain. All identifiers undergo **SHA-256 pseudonymization**.
2. **AES-256-GCM Patient Vault:** Diagnostic records and raw omics inputs are encrypted at rest using authenticated AES-256-GCM with PBKDF2 salt-derived keys (`src/bio_nexus/security/patient_vault.py`).
3. **Merkle Tree Batch Anchoring:** All 646 patient diagnostic fingerprints are structured into a binary SHA-256 Merkle Tree (`src/bio_nexus/security/merkle_tree.py`). A single 32-byte Merkle Root anchors the entire cohort on-chain in **one transaction ($O(1)$ gas cost)**.
4. **Instant Tamper Detection:** Verifying an individual patient requires only an $O(\log N)$ Merkle proof (10 SHA-256 hash operations, zero gas cost). Altering even a single gene expression value changes the leaf hash, invalidates the Merkle path, and immediately triggers a **TAMPER DETECTED** alert.

### Live On-Chain Proof (Polygon Amoy Testnet)

| Parameter | On-Chain Value |
|:---|:---|
| **Network** | **Polygon Amoy Testnet (Chain ID: 80002)** |
| **Smart Contract** | [`0xE771417C54c4968E7E324925a563a224BD16A30e`](https://amoy.polygonscan.com/address/0xE771417C54c4968E7E324925a563a224BD16A30e) |
| **Anchor Transaction** | [`0x9ac5e955a3809f504a7393c505bd85fe2c54114faba4a99519ba32d01768c1cd`](https://amoy.polygonscan.com/tx/0x9ac5e955a3809f504a7393c505bd85fe2c54114faba4a99519ba32d01768c1cd) |
| **Cohort Merkle Root** | `0x8d3e8633b1e0fc0d8e3cecddb19fed2e29e7111a488b4b9d1cac3b6871375ffd` |
| **Block Number** | `49,153,585` |
| **Patients Anchored** | **646 / 646 patients** in a single cryptographic batch |

---

## Technology Stack

| Domain | Technology / Library | Purpose |
|:---|:---|:---|
| **Deep Learning & Graph AI** | **PyTorch 2.3+** | Tensor computations, backpropagation, and custom modules |
| | **PyTorch Geometric (PyG) 2.5+** | GraphGPS layers (`GPSConv`), `GINEConv`, graph data structures |
| | **Captum 0.7+** | Multi-graph Integrated Gradients attribution computation |
| | **scikit-learn 1.4+** | Cosine similarity k-NN graph construction, stratified 5-fold CV |
| | **NetworkX 3.2+** | Graph topological metrics, degree distributions, purity analysis |
| **Backend & Serving** | **FastAPI 0.111+** | High-performance asynchronous REST API |
| | **Uvicorn 0.30+** | Production ASGI server |
| | **Pandas 2.2+ / NumPy 1.26+** | Omics matrix harmonization, log normalization, variance filtering |
| | **PyArrow 15.0+** | Columnar dataset ingestion and parquet serialization |
| | **Pydantic v2** | Request/response data contract enforcement |
| | **ReportLab 4.0+** | Automated 2-page publication-grade clinical diagnostic PDF generation |
| **Frontend & Visualization** | **React 19** | Modern reactive component architecture |
| | **TypeScript 5.0+** | Strict end-to-end type safety |
| | **Vite 8** | Lightning-fast development server and optimized bundle build |
| | **Tailwind CSS v4** | Modern dark-mode styling and fluid layout system |
| | **Recharts 3 + D3.js 7** | Dynamic scatter plots, radar charts, heatmaps, and interactive networks |
| | **Lucide Icons** | Consistent biomedical and clinical iconography |
| **Cryptography & Blockchain** | **Web3.py 6.18+** | Ethereum/Polygon RPC communication, transaction signing |
| | **eth-account** | Deterministic cryptographic keypair management |
| | **Solidity ^0.8.20** | `PatientRecordRegistry.sol` on-chain verification smart contract |
| | **Cryptography (Python)** | AES-256-GCM symmetric encryption with PBKDF2 key derivation |
| **MLOps & Infrastructure** | **MLflow 2.14+** | Experiment tracking, hyperparameter logging, model artifact registry |
| | **DVC (Data Version Control)** | Pipeline orchestration (`dvc.yaml`) for raw-to-processed stages |
| | **Docker** | Multi-stage container builds (`docker/Dockerfile.serving`) |
| | **Kubernetes** | Production deployment (`k8s/deployment.yaml`, `k8s/service.yaml`) |
| | **pytest** | Unit, model regression, and API contract test suites |

---

## Interactive Web Application

The BioNexus clinical suite (`frontend-new/`) provides oncologists, bioinformaticians, and pathology teams with an intuitive, production-grade interface:

| Page Route | View Name | Key Capabilities |
|:---|:---|:---|
| `/` | **Dashboard** | Executive overview: cohort statistics, model accuracy, service health, and recent patient diagnostic feeds. |
| `/patient-analysis` | **Patient Deep Dive** | Interactive patient inspection: predicted PAM50 subtype, calibrated confidence gauges, multi-omics modality contribution radar charts, top-20 gene Integrated Gradients waterfall, and live Polygon blockchain verification. |
| `/cohort` | **Cohort Analysis** | Interactive 2D UMAP and t-SNE projections showing unsupervised multi-omics patient clustering colored by PAM50 ground truth or clinical stage. |
| `/graph-explorer` | **Graph Explorer** | Interactive force-directed 2D/3D patient-similarity network displaying k-NN edges, node degrees, and inter-subtype boundary connections. |
| `/biomarkers` | **Biomarkers & Pathways** | Searchable table of all 5,000 genes, 1,881 miRNAs, and 3,957 CpG probes with Ensembl-to-HGNC mappings, global variance scores, and class-specific attribution rankings. |
| `/model-insights` | **Model Insights** | Full diagnostic suite: interactive out-of-fold confusion matrix, per-subtype ROC-AUC and PR curves, temperature calibration curves, and single-omics ablation comparisons. |
| `/live-case` | **Live Case Prediction** | Select real patient samples from the cohort, execute real-time model inference, inspect feature attributions, and verify cryptographic hashes. |
| `/new-case` | **New Case Intake** | Clinical specimen upload interface for new unseen patient profiles (RNA/miRNA/methylation vectors) with instant automated 2-page diagnostic PDF generation. |
| `/settings` | **Settings & Security** | API key management, Role-Based Access Control (Admin / Operator / Viewer), Polygon RPC status, and Merkle batch management. |

---

## REST API Reference

The FastAPI service runs on port `8000` (or configured via environment variables).

### Core Diagnostic Endpoints

| HTTP Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/health` | Liveness and readiness probe for container orchestrators. |
| `GET` | `/api/patients` | Returns list of all 646 cohort patients with true PAM50 labels. |
| `GET` | `/api/patients/{id}` | Detailed patient profile with normalized multi-omics values. |
| `POST` | `/api/predict` | Executes model inference returning predicted subtype, confidence, and modality weights. |
| `GET` | `/api/explain/{id}` | Returns Integrated Gradients attributions for top-ranking genes across omics layers. |
| `GET` | `/api/metrics` | Returns full validation metrics, confusion matrix, ablation benchmarks, and Wilcoxon statistics. |
| `GET` | `/v1/cases/{id}/report.pdf` | Generates and streams a 2-page publication-grade clinical molecular diagnostic PDF report. |
| `POST` | `/v1/panel-predict` | Fast CPU-friendly inference using a distilled 10-to-50 gene panel ($< 1.0\text{ ms}$ latency). |

### Blockchain Provenance Endpoints

| HTTP Method | Route | Description |
|:---|:---|:---|
| `GET` | `/api/security/cohort/status` | Current on-chain anchor status of the 646-patient cohort Merkle Root. |
| `POST` | `/api/security/cohort/anchor` | Anchors all 646 patient fingerprints to Polygon Amoy in a single transaction. |
| `GET` | `/api/security/cohort/verify/{id}` | Returns the cryptographic $O(\log N)$ Merkle proof and verifies tamper-evidence. |
| `GET` | `/api/security/patient/{id}` | Retrieves the individual on-chain audit receipt and PolygonScan explorer link. |

---

## Quick Start Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+ & npm**
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/bionexus-ml/bio_nexus.git
cd bio_nexus
```

### 2. Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Run the 5-Step Cohort Generation Pipeline
Execute the data engineering pipeline to construct the locked-in 646-patient cohort:
```bash
# Step 1: Intersect sample sheets across RNA-seq, miRNA, and Methylation
python3 scripts/patient_cohort.py

# Step 2: Build, normalize, and variance-filter omics matrices (HM450 selection)
python3 scripts/preprocess.py

# Step 3: Extract PAM50 clinical subtype labels
python3 scripts/extract_labels.py

# Step 4: Filter labels and conduct platform bias check
python3 scripts/finalize_labels.py

# Step 5: Lock in final 646-patient multi-omics cohort
python3 scripts/final_cohort.py
```

### 4. Start the FastAPI Backend
```bash
# Seed the metrics and graph statistics cache
python3 metrics_seed.py

# Launch the FastAPI inference server (runs on http://localhost:8000)
env PYTHONPATH=src uvicorn bio_nexus.serving.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Start the React 19 Frontend
In a new terminal window:
```bash
cd frontend-new
npm install
npm run dev -- --port 5174
```
Open [http://localhost:5174](http://localhost:5174) in your browser.

### 6. Running with Docker Compose (Optional)
```bash
docker-compose up --build
```

---

## Repository Layout

```
BioNexus/
├── BioNexus_Project_Writeup.md     Symlink to comprehensive scientific whitepaper
├── docs/
│   └── PROJECT_WRITEUP.md          Complete 569-line biological & ML specification
│
├── scripts/                        Reproducible Data Pipeline Scripts
│   ├── patient_cohort.py           Step 1 — Sample sheet parsing & omics intersection
│   ├── preprocess.py               Step 2 — Platform detection, log2 normalization & filtering
│   ├── extract_labels.py           Step 3 — PAM50 clinical subtype extraction
│   ├── finalize_labels.py          Step 4 — Subtype label filtering & platform bias verification
│   ├── final_cohort.py             Step 5 — Generation of locked-in 646-patient cohort
│   └── deploy_polygon.py           Smart contract deployment script for Polygon Amoy
│
├── contracts/                      Blockchain Smart Contracts
│   ├── PatientRecordRegistry.sol   Solidity contract for on-chain Merkle root anchoring
│   └── PatientRecordRegistry.json  Compiled EVM ABI
│
├── src/bio_nexus/                  Core Python Package
│   ├── data/                       Dataset ingestion, preprocessing, normalization
│   ├── graph/                      Patient-similarity k-NN graph construction
│   ├── encoding/                   GraphGPS hybrid encoder (GINEConv + Global Attention)
│   ├── fusion/                     Multimodal cross-attention fusion & learned pooling
│   ├── models/                     PAM50 classification head & temperature scaling
│   ├── interpretability/           Integrated Gradients & in-silico causal validation
│   │   ├── attribution.py          Multi-graph Integrated Gradients helper
│   │   ├── causal_ablation.py      Feature ablation & logit-impact calculation
│   │   └── statistical_validation.py Wilcoxon signed-rank paired hypothesis testing
│   ├── security/                   Cryptographic Provenance Layer
│   │   ├── patient_vault.py        AES-256-GCM encryption at rest + SHA-256 fingerprinting
│   │   ├── merkle_tree.py          Binary SHA-256 Merkle tree implementation
│   │   ├── cohort_manager.py       646-patient cohort batch Merkle manager
│   │   └── polygon_service.py      Polygon Amoy Web3 anchoring & verification
│   ├── serving/                    FastAPI serving layer
│   │   ├── api.py                  Core REST endpoints & routing
│   │   ├── ml_client.py            Inductive k-NN projection engine
│   │   └── report_pdf.py           Automated 2-page clinical PDF report generator
│   └── train.py                    Stratified 5-Fold Cross-Validation training loop
│
├── frontend-new/                   React 19 + TypeScript Clinical Web Application
│   ├── src/
│   │   ├── pages/                  Dashboard, Patient Analysis, Cohort, Graph Explorer, etc.
│   │   ├── components/             Reusable UI widgets, charts, and layout elements
│   │   ├── api/                    Typed API client for backend communication
│   │   └── lib/                    Types, utilities, and helper functions
│   ├── package.json
│   └── vite.config.ts
│
├── data/                           Data Store (DVC-tracked)
│   ├── raw/                        Raw TCGA-BRCA RNA, miRNA, and Methylation files
│   ├── processed/                  Normalized matrices, final cohort labels, graphs
│   └── security/                   Polygon anchor ledgers and Merkle batch records
│
├── configs/                        YAML experiment and training configurations
├── tests/                          Comprehensive pytest quality gate suite
├── docker/                         Multi-stage production Dockerfiles
├── k8s/                            Kubernetes deployment manifests
├── pyproject.toml                  PEP 518 / 621 Python project specification
└── dvc.yaml                        DVC pipeline specification
```

---

## Scientific References

1. **Parker, J. S. et al.** (2009). *Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes.* **Journal of Clinical Oncology**, 27(8), 1160–1167. [doi:10.1200/JCO.2008.18.1370](https://ascopubs.org/doi/10.1200/JCO.2008.18.1370)
2. **The Cancer Genome Atlas Network.** (2012). *Comprehensive molecular portraits of human breast tumours.* **Nature**, 490(7418), 61–70. [doi:10.1038/nature11412](https://www.nature.com/articles/nature11412)
3. **Rampášek, L. et al.** (2022). *Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS).* **NeurIPS 2022**, Advances in Neural Information Processing Systems. [arXiv:2205.12454](https://arxiv.org/abs/2205.12454)
4. **Sundararajan, M., Taly, A., & Yan, Q.** (2017). *Axiomatic Attribution for Deep Networks.* **ICML 2017**, PMLR 70:3319–3328. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
5. **Guo, C. et al.** (2017). *On Calibration of Modern Neural Networks.* **ICML 2017**, PMLR 70:1321–1330. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
6. **Jiang, Y. Z. et al.** (2019). *Genomic and Transcriptomic Landscape of Triple-Negative Breast Cancers: Subtypes and Treatment Strategies.* **Cancer Cell**, 35(3), 428–440. [doi:10.1016/j.ccell.2019.03.001](https://doi.org/10.1016/j.ccell.2019.03.001)

---

> **Clinical Disclaimer:** BioNexus is designed as an advanced computational biology and translational research platform. All molecular classifications and biomarker attributions are intended **For Research Use Only (RUO)** and must not be used as standalone clinical diagnostic or therapeutic guidance without validation by a qualified pathologist and clinical tumor board.
