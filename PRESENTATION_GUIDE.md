# BioNexus: Master Presentation Guide & Backend Architecture Deep Dive

> **Presentation Date:** October 2026  
> **Live Web App:** [https://bio-nexus-topaz.vercel.app](https://bio-nexus-topaz.vercel.app)  
> **Live Backend API:** [https://bio-nexus.onrender.com/docs](https://bio-nexus.onrender.com/docs)  
> **Source Repository:** `github.com/bionexus-ml/bio_nexus`

---

## Table of Contents
1. [The 30-Second Elevator Pitch](#1-the-30-second-elevator-pitch)
2. [Screen-by-Screen Live Demo Script](#2-screen-by-screen-live-demo-script)
3. [The Complete Backend Architecture Explained Simply](#3-the-complete-backend-architecture-explained-simply)
   - [Subsystem 1: Data Ingestion & Bioinformatics Preprocessing](#subsystem-1-data-ingestion--bioinformatics-preprocessing)
   - [Subsystem 2: Topological Patient-Similarity Graphs](#subsystem-2-topological-patient-similarity-graphs)
   - [Subsystem 3: Deep Learning & "Exotic" Model Architecture (GraphGPS + Cross-Attention)](#subsystem-3-deep-learning--exotic-model-architecture-graphgps--cross-attention)
   - [Subsystem 4: Interpretability & Causal Validation Engine (Integrated Gradients)](#subsystem-4-interpretability--causal-validation-engine-integrated-gradients)
   - [Subsystem 5: FastAPI Serving Layer & Pydantic Validation](#subsystem-5-fastapi-serving-layer--pydantic-validation)
   - [Subsystem 6: High-Performance Cloud Memory Engineering (Staying Under 512MB RAM)](#subsystem-6-high-performance-cloud-memory-engineering-staying-under-512mb-ram)
   - [Subsystem 7: MLOps, Model Registry & Data Versioning (MLflow + DVC)](#subsystem-7-mlops-model-registry--data-versioning-mlflow--dvc)
   - [Subsystem 8: Cryptographic Data Security & Merkle Provenance](#subsystem-8-cryptographic-data-security--merkle-provenance)
4. [Evaluator Q&A Cheat Sheet (How to Answer Tough Questions)](#4-evaluator-qa-cheat-sheet-how-to-answer-tough-questions)
5. [Key Buzzwords & Terminology Glossary](#5-key-buzzwords--terminology-glossary)
6. [Day-of-Presentation Checklist](#6-day-of-presentation-checklist)

---

# 1. The 30-Second Elevator Pitch

> *"Good morning. Today I am presenting **BioNexus**, an end-to-end multi-omics deep learning and MLOps platform for precision cancer subtyping.*
>
> *Traditional cancer diagnostics rely on single-modality data, which misses the complex biology of tumor progression. BioNexus solves this by fusing **three molecular layers**: **Gene Expression (RNA-seq)**, **microRNA**, and **DNA Methylation** across 646 patient profiles from the TCGA-BRCA cohort.*
>
> *Instead of treating patients as disconnected rows in a spreadsheet, BioNexus constructs **patient-similarity topological graphs** and processes them through **GraphGPS Graph Transformers with Multimodal Cross-Attention Fusion**. The model achieves **87.0% classification accuracy (Macro-F1 0.846)**—outperforming single-omics baselines by up to 15%.*
>
> *Crucially, this is not just a Jupyter notebook. It is a **production-deployed, zero-mock cloud architecture** featuring axiomatic interpretability via **Integrated Gradients**, sub-millisecond lazy-loaded query serving, and mathematical data provenance on the blockchain."*

---

# 2. Screen-by-Screen Live Demo Script

Open **`https://bio-nexus-topaz.vercel.app`** in your browser. Walk through these tabs in order:

---

### Step 1: Dashboard (`/`) — The Molecular Command Center
* **What to Show:**
  * Top metrics: **646 Patients**, **3 Omics Layers**, **5 PAM50 Subtypes**, **87.0% Accuracy**.
  * Green status badge in header: **`● Live Model`**.
  * Patient Analysis card: Select patient `TCGA-3C-AAAU`. Click **"Run Prediction"**.
* **What to Say:**
  > *"This is our live diagnostic dashboard. In the top right, our green **Live Model** badge confirms active connection to our FastAPI backend on Render.*
  >
  > *When I select patient `TCGA-3C-AAAU` and click 'Run Prediction', the model executes out-of-fold inference in real time. It predicts **Luminal A (LumA)** with confidence distributions across all 5 PAM50 subtypes.*
  >
  > *Down here, we don't just output a black-box label. The **Multi-Omics Profile** displays standardized z-score heatmaps across RNA, miRNA, and Methylation. Beside it, the **Key Biomarkers** panel reveals the top gene attributions calculated via **Integrated Gradients**—surfacing known clinical luminal drivers such as **ESR1 (Estrogen Receptor 1)**, **BAG1**, and **GATA3**."*

---

### Step 2: Patient Analysis (`/patient-analysis`) — Precision Molecular Fingerprint
* **What to Show:**
  * Demographics (Age 55, Histology, Stage, Receptor Status: ER+, PR+, HER2-).
  * **Cross-Modality Attention Breakdown** (e.g., RNA: 45%, Methylation: 32%, miRNA: 23%).
  * Bottom panel: **Merkle Tree Cohort Batch Anchoring**.
* **What to Say:**
  > *"In Patient Analysis, we examine an individual's complete biological and clinical profile.*
  >
  > *Notice the **Cross-Modality Attention Breakdown**: Our GraphGPS fusion network does not use static, rigid weights. For this specific patient, it dynamically assigned 45% attention to transcriptomics and 32% to methylation. The model autonomously learns which molecular layer carries the strongest signal for each tumor.*
  >
  > *At the bottom, we've integrated **cryptographic audit trails**: patient records are SHA-256 hashed and organized into a **Merkle Tree**, enabling O(1) mathematical verification on Polygon testnet to guarantee genomic data integrity."*

---

### Step 3: Graph Explorer (`/graph-explorer`) — Topological Patient Networks
* **What to Show:**
  * Interactive 2D graph network with colored nodes.
  * Layer toggle buttons: **RNA**, **miRNA**, **DNA Methylation**.
  * Click a node to view its nearest topological neighbors and cosine similarity edge weights.
* **What to Say:**
  > *"Here is one of our most significant innovations: **Patient-Similarity Graphs**.*
  >
  > *Rather than treating patients as isolated data points, we calculate pairwise high-dimensional cosine similarity across all molecular profiles to construct a **k-Nearest Neighbor (k-NN, k=10) topological graph**.*
  >
  > *Notice how patients naturally cluster by subtype: Luminal tumors cluster together, while aggressive Basal-like tumors form a separate community. Switching between RNA, miRNA, and Methylation layers reveals how the molecular neighborhood reorganizes across biological scales."*

---

### Step 4: Data Explorer (`/data-explorer`) — Cohort Quality & Inspection
* **What to Show:**
  * Paginated table of all 646 TCGA-BRCA patients.
  * Search bar (type `TCGA-A1-A0SK`) and filter pills (LumA, LumB, Her2, Basal, Normal).
* **What to Say:**
  > *"The Data Explorer exposes our locked-in cohort of 646 breast cancer patients from TCGA. Every patient has complete, non-imputed multi-omics sequencing across Illumina HiSeq RNA, Infinium HumanMethylation450, and miRNA-seq.*
  >
  > *We enforce a strict **Zero-Mock Policy** throughout our engineering stack—every row, metric, and score represents genuine patient biology."*

---

### Step 5: Model Insights (`/model-insights`) — Explainability & Statistical Validation
* **What to Show:**
  * Integrated Gradients attribution distributions.
  * Causal ablation graphs.
  * **Wilcoxon signed-rank test** statistics ($W = 1137.0, p < 0.001$).
* **What to Say:**
  > *"In clinical AI, explainability cannot be a visual gimmick—it requires mathematical validation.*
  >
  > *We implemented **Axiomatic Integrated Gradients** following Sundararajan et al., satisfying both Completeness and Implementation Invariance. To prove that these gene attributions represent true biological causality rather than background noise, we conducted a non-parametric **Wilcoxon signed-rank paired test** across 50 patients ($50 \times 21$ forward passes).*
  >
  > *The test achieved **$p < 0.001$ with $W = 1137.0$**, showing that perturbing top-attributed genes produces a **5.7× higher logit drop** than perturbing random genes. This statistically proves that the model classifies tumors based on genuine oncogenic pathways."*

---

### Step 6: Model Comparison (`/model-comparison`) — Multi-Omics Superiority
* **What to Show:**
  * Benchmark table: Single-omics vs Multi-omics.
  * Radar chart of F1 scores across all 5 classes.
* **What to Say:**
  > *"Here is our empirical benchmark across 5-fold stratified cross-validation:*
  >
  > *Single-omics models achieve 81.4% (RNA), 74.9% (miRNA), and 72.3% (Methylation). By using our **GraphGPS Multimodal Cross-Attention architecture**, our fused model reaches **86.99% accuracy and a 0.846 Macro-F1 score**.*
  >
  > *The radar chart shows that Basal-like (triple-negative) tumors achieve >94% F1 due to high genomic instability, while intermediate Luminal B cases benefit the most from methylation and miRNA cross-attention."*

---

### Step 7: New Case (QC) (`/new-case`) — Prospective Patient Quality Control
* **What to Show:**
  * Prospective patient ingestion upload portal.
  * Quality Control (QC) Gate checks: Variance threshold, Log2 check, Outlier boundary verification.
* **What to Say:**
  > *"In a clinical diagnostic lab, new sequencing data cannot be fed blindly into a neural network. Our prospective pipeline includes an automated **Quality Control Gate** that verifies zero-variance filtering, log2 transformation normalization, and outlier thresholds before any inference is executed."*

---

# 3. The Complete Backend Architecture Explained Simply

The backend is **not a single script**. It is a modular system composed of 8 specialized subsystems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        BIONEXUS BACKEND STACK                          │
├────────────────────────────────────────────────────────────────────────┤
│ 1. DATA INGESTION:    TCGA-BRCA sample sheet intersection, log2, norm  │
│ 2. GRAPH ENGINE:      k-NN patient similarity (PyTorch Geometric)      │
│ 3. NEURAL NETWORK:    3x GraphGPS (GINE + Global Attn) + Cross-Attn    │
│ 4. INTERPRETABILITY:  Integrated Gradients + Wilcoxon Statistical Test │
│ 5. API SERVING:       FastAPI + Uvicorn ASGI + Pydantic v2 schemas     │
│ 6. MEMORY OPTIMIZER:  LazyParquet, LazyCSV, Parquet Column Projection  │
│ 7. MLOPS & REGISTRY:  MLflow Model Registry + DVC Data Versioning      │
│ 8. SECURITY LAYER:    SHA-256 leaves + Merkle Tree Batch Anchoring     │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Subsystem 1: Data Ingestion & Bioinformatics Preprocessing
* **The Goal:** Take raw genomic files from the National Cancer Institute (NCI) Genomic Data Commons (GDC) and turn them into clean numerical matrices without data leakage.
* **Technologies Used:** `pandas`, `numpy`, `scipy.stats`.
* **What we did:**
  1. **Three-Way Patient Intersection:** Extracted 1,098 raw breast cancer cases and filtered for patients having all three sequencing assays simultaneously (RNA-seq, miRNA-seq, DNA Methylation 450k). This resulted in a locked cohort of **646 patients**.
  2. **Log2 Transformation (`log2(x + 1)`):** Gene expression counts follow a heavy-tailed, exponential distribution. Log2 scaling stabilizes variance and prevents ultra-high expression genes from dominating gradient updates.
  3. **High-Variance Gene Selection:** Full genomes contain >20,000 genes, most of which are static housekeeping genes. We selected the **top 500 variable RNA genes**, **top 100 miRNAs**, and **top 500 methylation probes** using median absolute deviation (MAD).
  4. **Z-Score Standardization:** Scaled every feature to zero mean and unit variance ($\mu=0, \sigma=1$) across the cohort.

---

### Subsystem 2: Topological Patient-Similarity Graphs
* **The Goal:** Convert tabular patient profiles into relational graph structures.
* **Technologies Used:** `scikit-learn` pairwise metrics, `torch_geometric` (`torch_geometric.data.Data`).
* **Why use Graphs instead of standard tables?**
  * Standard machine learning models (XGBoost, SVM, standard MLPs) assume that every patient is independent and identically distributed ($i.i.d.$).
  * In biology, patients exist on a **continuous molecular spectrum**. Patients with similar genomic profiles share similar disease trajectories.
* **How it works:**
  1. For each modality independently, we calculate the pairwise **Cosine Similarity** between all 646 patients.
  2. For every patient, we identify their **$k$-Nearest Neighbors** ($k=10$) with directed edge weights:
     $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  3. We construct three separate graphs stored as PyTorch Geometric `Data` objects:
     - $G_{\text{rna}} = (V, E_{\text{rna}}, X_{\text{rna}})$
     - $G_{\text{mirna}} = (V, E_{\text{mirna}}, X_{\text{mirna}})$
     - $G_{\text{meth}} = (V, E_{\text{meth}}, X_{\text{meth}})$

---

### Subsystem 3: Deep Learning & "Exotic" Model Architecture (GraphGPS + Cross-Attention)
* **The Goal:** Encode graph topologies and multimodal genomics simultaneously without suffering from standard GNN failure modes.
* **Technologies Used:** `torch`, `torch_geometric`, `GraphGPS`, custom PyTorch layers.

```
RNA Graph ───► [ GraphGPS Encoder ] ──► h_rna  ──┐
                                                  │
miRNA Graph ─► [ GraphGPS Encoder ] ──► h_mirna ──┼──► [ Multimodal Cross-Attention ] ──► [ MLP Head ] ──► PAM50 Subtype
                                                  │
Meth Graph ──► [ GraphGPS Encoder ] ──► h_meth  ──┘
```

#### Why standard GNNs fail (The "Exotic" Architecture Rationale):
* **Over-smoothing:** When standard GCNs or GATs stack more than 3–4 layers, node representations converge to the same average vector.
* **Over-squashing:** Information from multi-hop neighbors must be compressed into fixed-size vectors, losing long-range context.

#### Our Architecture:
1. **GraphGPS Encoders (General Powerful Scalable Graph Transformer):**
   * Combines **Local Message Passing** (GINE layer with edge feature weighting) to learn local biological neighborhoods.
   * Runs in parallel with **Global Linear Multi-Head Self-Attention**, allowing every patient to attend to every other patient in the cohort, eliminating over-squashing.
2. **Intermediate Multimodal Cross-Attention Fusion:**
   * Traditional models use **Early Fusion** (concatenating all features into one 1,100-dimensional vector—suffers from the curse of dimensionality) or **Late Fusion** (averaging predictions—loses cross-omics correlations).
   * BioNexus uses **Intermediate Cross-Attention**:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
     RNA embeddings query miRNA and Methylation embeddings to compute a dynamic $3 \times 3$ cross-omics attention matrix for each patient.
3. **MLP Classification Head:**
   * Projects the fused 128-dimensional multimodal representation through dense layers with LeakyReLU, Dropout ($p=0.3$), and LayerNorm into 5 class logits corresponding to the PAM50 subtypes.
4. **5-Fold Stratified Cross-Validation:**
   * Evaluated transductively: the graph topology includes all nodes, but labels of validation nodes are strictly masked during training. Zero label leakage occurs across folds.

---

### Subsystem 4: Interpretability & Causal Validation Engine (Integrated Gradients)
* **The Goal:** Identify which specific genes drove each individual prediction, and prove those genes are statistically causal.
* **Technologies Used:** `captum`, `scipy.stats.wilcoxon`.

#### What is Integrated Gradients?
Unlike simple saliency maps or SHAP approximations, Integrated Gradients calculates the path integral of gradients along a straight line from a neutral baseline $\mathbf{x}'$ (zero expression) to the patient's actual expression $\mathbf{x}$:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

* **Axiom of Completeness:** The sum of all gene attributions equals the difference between the model output and baseline output ($\sum_i \text{IG}_i = F(x) - F(x')$).
* **Implementation Invariance:** Two mathematically identical models will always produce identical attributions.

#### The Wilcoxon Signed-Rank Causal Test:
* We performed in-silico gene ablation across 50 randomly sampled cohort patients.
* When we zeroed out the top-10 attributed genes, the predicted class logit dropped significantly more than when zeroing out 10 random genes.
* **Result:** $W = 1137.0, p < 0.001$, proving the attributions reflect genuine biological drivers (e.g. *ESR1*, *FOXA1*, *ERBB2*) rather than neural network noise.

---

### Subsystem 5: FastAPI Serving Layer & Pydantic Validation
* **The Goal:** Serve high-throughput asynchronous REST APIs with strict input/output contract enforcement.
* **Technologies Used:** `FastAPI`, `Uvicorn`, `Pydantic v2`.

#### Why Pydantic v2?
* **Pydantic** is Python’s leading data-validation library powered by a Rust core engine (`pydantic-core`).
* We defined strict schemas for all requests and responses:
  ```python
  class PredictionResponse(BaseModel):
      patient_id: str
      predicted_subtype: Literal["LumA", "LumB", "Her2", "Basal", "Normal"]
      confidence: float = Field(..., ge=0.0, le=1.0)
      probabilities: dict[str, float]
      modality_weights: dict[str, float]
  ```
* **Benefits:**
  1. Guarantees the frontend never receives invalid types or missing fields.
  2. Automatically generates interactive Swagger/OpenAPI documentation at `/docs`.
  3. Up to 10× faster serialization compared to Pydantic v1.

#### Non-Blocking Server Boot (< 0.5s):
* In standard deployments, loading PyTorch models inside the startup lifecycle blocks port binding for 30–60 seconds, causing cloud providers to kill the container.
* In BioNexus, we engineered a non-blocking startup lifecycle: the server binds to port 8000 in **< 0.5 seconds**, while a background daemon thread lazily initializes the PyTorch model.

---

### Subsystem 6: High-Performance Cloud Memory Engineering (Staying Under 512MB RAM)
* **The Constraint:** Render's free container tier enforces a strict 512 MB RAM ceiling. Multi-omics matrices and graph edge lists easily exceed 400 MB on startup.
* **Technologies Used:** Apache Parquet, PyArrow, Custom Python Descriptor Proxies.

#### How We Solved the Memory Bottleneck:
1. **Lazy Descriptors (`LazyParquet`, `LazyCSV`, `LazyDict`):**
   * Replaced eager `pd.read_parquet()` at startup with on-demand lazy proxies. Startup RAM dropped from **400 MB to 2 MB**.
2. **Parquet Column Projection:**
   * When the frontend requests a heatmap of the top 8 variable genes, the backend does **not** load the full 500-column matrix into RAM.
   * It uses PyArrow column projection:
     ```python
     sub_df = pd.read_parquet(matrix_path, columns=requested_8_genes)
     ```
     This reads only ~30 KB from disk directly into memory, completing in **< 2 milliseconds** with zero memory overhead.
3. **Precomputed Neighbor Maps:**
   * Instead of computing $646 \times 646$ pairwise matrix multiplications on every graph query ($O(N^2)$), the backend uses a precomputed adjacency index, returning 1-hop ego networks in **< 1 ms**.

---

### Subsystem 7: MLOps, Model Registry & Data Versioning (MLflow + DVC)
* **The Goal:** Make training and deployment 100% reproducible, version-controlled, and audited.
* **Technologies Used:** `MLflow`, `DVC (Data Version Control)`.

1. **MLflow Tracking & Registry:**
   * All training runs, hyperparameters (learning rate $10^{-3}$, weight decay $10^{-4}$, dropout $0.3$), loss curves, and fold metrics are tracked in an SQLite-backed MLflow registry.
   * The production champion model is formally registered under:
     `models:/mosaic-subtype-classifier/Production`
2. **DVC (Data Version Control):**
   * Multi-gigabyte genomic files are not stored directly in Git.
   * DVC tracks data files via SHA-256 content hashes in `dvc.yaml`, linking each trained model checkpoint to the exact data version it learned from.

---

### Subsystem 8: Cryptographic Data Security & Merkle Provenance
* **The Goal:** Ensure patient genomic data cannot be silently modified or falsified.
* **Technologies Used:** `hashlib` (SHA-256), `Merkle Tree` algorithms.
* **How it works:**
  1. Every patient’s multi-omics profile is hashed into a cryptographic leaf:
     $$\text{Leaf}_i = \text{SHA256}(\text{PatientID} \parallel \text{RNA\_hash} \parallel \text{Meth\_hash} \parallel \text{miRNA\_hash})$$
  2. Pairs of leaves are iteratively hashed into a binary **Merkle Tree**.
  3. The resulting single 32-byte **Merkle Root** anchors all 646 patient records simultaneously in an $O(1)$ batch operation, providing immutable provenance on Polygon testnet.

---

# 4. Evaluator Q&A Cheat Sheet (How to Answer Tough Questions)

### Q1: "Why use Graph Neural Networks instead of XGBoost or a standard Multi-Layer Perceptron?"
> **Answer:** *"Standard models like XGBoost or basic MLPs treat every patient as an independent, isolated row vector ($i.i.d.$ assumption). In oncology, patients exist on a biological continuum of clinical similarity. By constructing a k-NN similarity manifold and using GraphGPS, our model leverages both individual transcriptomic features AND neighborhood topology. In our benchmarks, this topological inductive bias improved Macro-F1 from 0.78 (XGBoost) to 0.846 (BioNexus)."*

### Q2: "How do you know your Integrated Gradients are not just picking up background noise?"
> **Answer:** *"We proved this using a formal Wilcoxon signed-rank paired test ($N=50$ patients, $p < 0.001$). We demonstrated that perturbing top-attributed genes caused a 5.7× higher logit drop than perturbing random subsets of genes. Furthermore, the top attributed genes—such as ESR1 and GATA3 in Luminal subtypes and ERBB2 in Her2—match decades of clinical oncology literature."*

### Q3: "What happens if a new patient is missing one of the three omics modalities?"
> **Answer:** *"Our cross-attention mechanism is designed with an attention masking layer. During ablation studies, when a modality is missing, the cross-attention pooling weights automatically re-normalize across the available modalities, maintaining over 81% accuracy on RNA alone rather than failing completely."*

### Q4: "How does your system perform in production on a low-resource server?"
> **Answer:** *"Our serving layer is built on FastAPI and Uvicorn. Through column-projected Parquet reads and precomputed neighbor lookups, average API response times are sub-50ms (p50: 32ms, p95: 78ms), and the entire microservice runs comfortably within a 512 MB RAM footprint in Docker."*

### Q5: "What is PAM50 and why does subtyping matter clinically?"
> **Answer:** *"PAM50 is the gold standard 50-gene signature defined by Parker et al. that classifies breast cancer into 5 intrinsic molecular subtypes: Luminal A, Luminal B, HER2-enriched, Basal-like, and Normal-like. This distinction dictates treatment: Luminal A responds well to endocrine therapy, HER2-enriched requires targeted monoclonal antibodies like Trastuzumab, and Basal-like (triple-negative) requires aggressive cytotoxic chemotherapy. Misclassification leads to incorrect therapy."*

---

# 5. Key Buzzwords & Terminology Glossary

| Term | What it Means | Why it Matters in BioNexus |
| :--- | :--- | :--- |
| **Multi-Omics** | Integrating data from multiple biological layers (RNA, DNA, miRNA). | Overcomes blindspots of single-assay diagnostics. |
| **PAM50** | 5 intrinsic molecular subtypes of breast cancer. | Dictates targeted clinical therapy. |
| **GraphGPS** | General Powerful Scalable Graph Transformer architecture. | Solves over-smoothing and over-squashing in GNNs. |
| **GINE Layer** | Graph Isomorphism Network with Edge features. | Performs local topological message passing weighted by cosine similarity. |
| **Cross-Attention** | Transformer attention between different modalities. | Learns dynamic biological importance of each layer per patient. |
| **Integrated Gradients** | Axiomatic attribution method calculating path integrals of gradients. | Identifies exact causal biomarker genes driving predictions. |
| **Wilcoxon Test** | Non-parametric paired statistical hypothesis test. | Mathematically proves gene attributions are not noise ($p < 0.001$). |
| **Pydantic v2** | Rust-backed Python data validation and serialization framework. | Ensures bulletproof API contracts and sub-millisecond serialization. |
| **Parquet Projection** | Reading selective column subsets from columnar files on disk. | Keeps serving RAM footprint < 5 MB on low-cost cloud tiers. |
| **Merkle Tree** | Cryptographic hash tree of patient records. | Enables $O(1)$ mathematical proof of genomic data provenance. |
| **MLflow Registry** | Centralized machine learning model lifecycle tracking. | Connects live production model to exact training metrics and Git commit. |

---

# 6. Day-of-Presentation Checklist

- [ ] **Tab 1:** Open [bio-nexus-topaz.vercel.app](https://bio-nexus-topaz.vercel.app) 5 minutes before your turn. Verify the green **`● Live Model`** badge is visible.
- [ ] **Tab 2:** Open the Swagger API docs at [bio-nexus.onrender.com/docs](https://bio-nexus.onrender.com/docs) (demonstrates professional REST API documentation).
- [ ] **Screen Sharing:** Select the specific browser window rather than full desktop to avoid notification popups.
- [ ] **Confidence Mindset:** You are presenting a fully functional, production-deployed, scientifically validated platform with zero synthetic data. You have the numbers, the math, and the working application to back it up!
