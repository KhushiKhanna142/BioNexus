# BioNexus: Multi-Omics Graph Transformer for Breast Cancer Subtype Classification

## 1. Motivation

Breast cancer is molecularly heterogeneous. The PAM50 classification system (Parker et al., 2009, *Journal of Clinical Oncology*) established that gene-expression-based molecular subtyping — LumA, LumB, Her2-enriched, Basal-like, Normal-like — predicts patient outcome independently of standard clinical markers (ER status, tumor grade, lymph node status), and has since become a clinically deployed diagnostic test (Prosigna). PAM50 subtype is the prediction target for this project.

The value of integrating multiple molecular layers — rather than relying on gene expression alone — was established at scale by The Cancer Genome Atlas Network's foundational analysis (2012, *Nature*), which profiled 825 breast tumors across six platforms (DNA methylation, mRNA, miRNA, protein expression, copy number, mutation) and found that different omics layers capture distinct, non-redundant biological signal: subgroups distinguishable via protein/mRNA expression were indistinguishable using methylation, mutation, or copy-number data alone, and vice versa. This motivates multi-omics integration as a principled modeling requirement rather than an empirical convenience, and is the dataset this project's cohort is drawn from (TCGA-BRCA).

Real-world clinical relevance: multi-omics-derived subtyping has already informed treatment decisions in hospital settings — e.g. Fudan University Shanghai Cancer Center's 465-patient multi-omics TNBC classification (Jiang et al., 2019), which identified a luminal androgen receptor subgroup subsequently shown to respond to anti-androgen therapy.

## 2. Data

**Source:** TCGA-BRCA (The Cancer Genome Atlas – Breast Invasive Carcinoma), downloaded from the GDC Data Portal.

**Three omics layers, patient-matched, primary tumor samples:**
- **RNA-seq:** GDC STAR-Counts pipeline, gene-level TPM expression, ~60,660 genes (Ensembl IDs)
- **miRNA:** GDC miRNA quantification, ~1,881 miRNAs, read counts
- **DNA methylation:** Beta values (0–1 scale). Discovered to span two array platforms — Illumina HumanMethylation450 (~486,427 probes, majority) and the older HumanMethylation27 (~27,578 probes, minority) — which cannot be merged due to differing probe sets

**Labels:** PAM50 molecular subtype (LumA, LumB, Her2, Basal, Normal), extracted from cBioPortal's TCGA-BRCA PanCancer Atlas clinical export.

## 3. Cohort Construction Pipeline

| Stage | Script | Outcome |
|---|---|---|
| Patient intersection | `patient_cohort.py` | Intersected patients present in all 3 omics layers → 1,035 patients |
| Label extraction | `extract_labels.py` | Merged PAM50 labels from cBioPortal; 104/1,035 patients had no label (not all TCGA patients receive a confident PAM50 call) |
| Preprocessing | `preprocess.py` | Built normalized per-layer matrices; discovered methylation platform split, kept dominant HM450 platform only → cohort reduced to 733 patients |
| Bias check + label reconciliation | `finalize_labels.py` | Verified the 302 dropped (HM27-platform) patients did not skew subtype distribution (largest gap: 6.3 percentage points, Her2) — platform drop is not a source of bias. 87/733 remaining patients had no subtype label |
| Final cohort lock-in | `finalize_cohort.py` | **Final training cohort: 646 patients**, all 3 omics layers + valid PAM50 label, aligned across matrices |

## 4. Preprocessing Details

- **RNA-seq:** log2(x+1) normalized, then variance-filtered to top 5,000 most-variable genes (added during diagnostic fixes — see Section 7)
- **miRNA:** log2(x+1) normalized, no dimensionality reduction needed (1,881 features)
- **Methylation:** already 0–1 scaled (no transform), variance-filtered to top 5,000 probes (from 486,427) before missing-value cleanup for computational tractability; 3,957 remained after dropping high-missingness probes
- **Missing values:** features missing in >20% of patients dropped; remainder median-imputed

## 5. Model Architecture

**Design rationale:** prior multi-omics GNN work for cancer subtyping (MOGONET, GATOmics, FaGGCN) relies on local graph convolution/attention (GCN/GAT), which aggregates only over a node's immediate neighbors. This project uses a **graph transformer** architecture, which combines local message-passing with *global* attention across the full patient graph.

### 5.1 Graph Construction (`graph/construction.py`)
- One k-NN patient-similarity graph per omics layer (patients as nodes, not genes/molecules — distinguishing this from graph-based approaches using molecule-interaction graphs, e.g. Kaczmarek et al., 2022)
- k=10 nearest neighbors, cosine similarity, edges symmetrized
- Node order verified identical across all 3 layer graphs (required for later fusion)
- Result: 646 nodes per layer; RNA ~11,356 edges, miRNA ~11,168 edges, methylation ~12,114 edges

### 5.2 GraphGPS Encoding (`encoding/graph_transformer.py`)
- **GraphGPS** (Rampášek et al., NeurIPS 2022) — a hybrid graph transformer combining local edge-aware message-passing (GINE convolution) with global multi-head self-attention in each layer
- One encoder per omics layer (separate weights, different input dimensionality per layer)
- Output: 64-dimensional embedding per patient, per layer, with output LayerNorm (added during diagnostic fixes) ensuring comparable scale across modalities before fusion

### 5.3 Cross-Attention Fusion (`fusion/cross_attention.py`)
- Learned multi-head self-attention across the 3 modality embeddings per patient (each tagged with a learnable modality-identity embedding)
- Learned softmax pooling combines the 3 attended modality representations into one fused embedding per patient
- Pooling weights double as an interpretability signal: per-patient, per-modality contribution to the final prediction

### 5.4 Classification Head (`models/classifier.py`)
- Fused embedding → 2-layer MLP → 5-way softmax (PAM50 subtype)
- Full model: **~4.68M parameters**

## 6. Training Methodology (`train.py`)

- **Transductive setting:** all 646 patients exist as nodes in one shared graph throughout training; cross-validation masks which nodes' *labels* are visible during training vs. evaluation (node features, including validation-fold patients', remain visible throughout — standard practice for GNN evaluation, not label leakage)
- **Stratified 5-fold cross-validation** (preserves subtype proportions per fold, given class imbalance: LumA ~50%, Normal ~4.5%)
- **Class-weighted loss** (computed from training-fold labels only, per fold)
- **Early stopping** on validation macro-F1 (not accuracy, given class imbalance)
- **AdamW with weight decay** for regularization, given the parameter-to-patient ratio (~4.68M params / 646 patients)

## 7. Ablation Study and Diagnostic Investigation

To directly test whether multi-omics fusion outperforms any single omics layer for this architecture — rather than assuming it does, as most prior work does without testing — a single-omics ablation was run (`evaluation/ablation.py`), training the same encoder + classifier architecture on each omics layer independently, using identical CV folds for fair comparison.

### 7.1 Initial Finding (before diagnostic fixes)

| Layer | Accuracy | Macro F1 |
|---|---|---|
| RNA alone | 61.9% ± 14.5% | 50.4% ± 17.4% |
| miRNA alone | **77.9% ± 3.9%** | **71.9% ± 4.4%** |
| Methylation alone | 71.8% ± 7.6% | 69.8% ± 10.1% |
| Fused (all 3) | 73.9% ± 9.0% | 71.1% ± 7.5% |

**Fusion underperformed the best single-omics layer (miRNA alone).** This counterintuitive result was investigated rather than dismissed.

### 7.2 Diagnosis (`evaluation/diagnose_fusion.py`)

Inspecting the trained fusion module's learned pooling weights revealed the root cause: fusion had learned to weight **methylation at 64%** on average — the *dominant* modality — despite methylation being the weaker of the two competitive single-omics layers (miRNA outperformed it). Two architectural causes were identified:

1. **RNA lacked dimensionality reduction.** All 60,660 raw genes fed a single linear projection with only 646 training patients — a severely underdetermined projection likely producing noisy embeddings, consistent with RNA's poor standalone performance.
2. **No normalization on encoder outputs before fusion.** Each of the 3 encoders trained with independent weights and no mechanism forced their output embeddings onto a comparable scale — the fusion pooling layer could therefore be influenced by incidental embedding magnitude rather than purely predictive relevance.

### 7.3 Fixes Applied

1. RNA variance-filtered to top 5,000 genes (post-log2-normalization, matching methylation's approach)
2. `LayerNorm` added to each encoder's final output, before fusion

### 7.4 Result After Fixes

| Layer | Accuracy | Macro F1 |
|---|---|---|
| RNA alone | **81.4% ± 8.4%** | 81.0% ± 7.7% |
| miRNA alone | 74.9% ± 4.1% | 72.4% ± 3.6% |
| Methylation alone | 72.3% ± 3.2% | 72.8% ± 3.7% |
| **Fused (all 3)** | **87.0% ± 3.0%** | **84.6% ± 3.0%** |

Fusion now decisively outperforms every single-omics layer, including the now-strongest RNA-alone baseline (+5.5 points accuracy, +3.5 points macro-F1). Fold-to-fold variance also dropped sharply (accuracy std: ±9.0% → ±3.0%), indicating the fixes improved training stability, not just the average.

**Confusion matrix (best fold, macro-F1 = 0.882):**

```
              Predicted
              LumA  LumB  Her2  Basal Normal
True LumA      62    3     1     0     2
True LumB       6   20     0     0     0
True Her2       0    0     7     0     0
True Basal      0    0     0    23     0
True Normal     0    0     0     1     5
```

Per-class recall: LumA 91.2%, LumB 76.9%, Her2 100%, Basal 100%, Normal 83.3%. The LumA/LumB boundary remains the hardest distinction — consistent with known clinical difficulty in PAM50 subtyping generally, not an artifact specific to this model.

**Interpretability note:** RNA's fusion pooling weight is notably higher on incorrect predictions (mean 0.77) than correct ones (mean 0.61) — suggesting over-reliance on a single modality, even the strongest one, correlates with reduced reliability, while balanced multi-modal weighting correlates with correct predictions.

## 8. Positioning Relative to Prior Work

| Work | Graph unit | Architecture | Fusion | Omics layers |
|---|---|---|---|---|
| MOGONET | Patient | GCN | View Correlation Discovery Network (VCDN) | 3 |
| GATOmics | Patient | GAT (local attention only) | — | varies |
| MOGAT | Patient | GAT (local attention only) | VCDN / KD-SVAE-VCDN | 3 |
| Kaczmarek et al. (2022) | Molecule (miRNA–mRNA targeting) | Graph Transformer | — | 2 |
| **BioNexus (this work)** | **Patient** | **GraphGPS (hybrid local + global attention)** | **Learned cross-attention** | **3** |

The closest prior work is MOGAT, which shares the patient-similarity-graph design but uses GAT (local-neighbor attention only, no global attention) and VCDN-style fusion rather than learned cross-attention. This project's distinguishing contribution is the combination of (1) global-attention graph transformer encoding instead of local-only GAT, and (2) learned, per-patient interpretable cross-attention fusion instead of a fixed correlation-based fusion module — plus an explicit, documented ablation-and-diagnosis methodology validating that fusion provides real benefit for this architecture, addressing a gap (causal ablation validation) identified in the literature review.

## 9. Interpretability

Two complementary interpretability mechanisms were built, both requiring a trained model to be meaningful (untrained embeddings would produce uninterpretable noise):

### 9.1 Gradient-Based Attribution (`interpretability/attribution.py`)

Integrated Gradients, implemented directly rather than via a general-purpose library — the model takes 3 separate graph inputs sharing fixed graph structure, so attribution interpolates one layer's raw features between a zero baseline and the real input (50 steps), holding the other two modalities and all graph structure fixed, then integrates the gradient of the target patient's predicted-class logit along that path. Produces a per-gene attribution score for a given patient, identifying which genes pushed the prediction toward or away from the predicted class.

### 9.2 Causal Validation of Attribution (`interpretability/causal_ablation.py`, `interpretability/statistical_validation.py`)

Rather than assuming attribution scores reflect genuinely causal importance, this was tested directly: the top-20 attribution-flagged genes for a patient were ablated (zeroed out) and the resulting change in the model's prediction was measured, compared against ablating 20 randomly selected genes as a control.

**Methodological note:** an initial check using softmax-probability drop was found to be misleading — most patients' predictions were already at 95–99% confidence, a saturated regime where probability barely moves regardless of which features are removed, producing unstable ratios (including two apparently large but meaningless ratios computed from near-zero values on both sides). This was corrected by measuring the change in absolute logit value (pre-softmax), which does not saturate the same way.

**Statistical validation:** across 50 patients, a paired Wilcoxon signed-rank test compared the absolute logit-impact of ablating top-attribution genes against ablating random genes:

- Median top-K attribution impact: 0.0045 (vs. median random impact: 0.0010 — ~4.5x larger)
- Mean top-K attribution impact: 0.0246 (vs. mean random impact: 0.0043 — ~5.7x larger)
- Attribution-flagged genes outperformed random gene selection in **41/50 patients (82%)**
- **Wilcoxon signed-rank test: p < 0.001**

This provides statistically significant, causal (not merely correlational) evidence that gradient-based attribution identifies genes the model genuinely relies on for its predictions — directly addressing the "causal ablation validation" gap identified in the literature review (Section 8), where prior multi-omics GNN papers typically report attention weights as interpretability without validating that attended features are causally responsible for predictions.

**Secondary finding:** RNA's fusion pooling weight is notably higher on incorrect predictions (mean 0.77) than correct ones (mean 0.61) — over-reliance on a single modality, even the strongest one, correlates with reduced reliability, while balanced multi-modal weighting correlates with correct predictions (see Section 7.4).

## 10. Engineering / MLOps

Repository restructured into a production-style package (`src/bio_nexus/`) with modules for data, graph, encoding, fusion, models, interpretability, evaluation, registry, serving, and utils. MLflow experiment tracking, Docker containerization (dev/train/serving), FastAPI inference serving, and CI/CD (GitHub Actions) integrated in collaboration with a teammate handling MLOps.

## 11. Current Status and Remaining Work

**Completed:** cohort construction, preprocessing, graph construction, GraphGPS encoding, cross-attention fusion, classifier, training loop with stratified CV, single-omics ablation study, diagnostic investigation and fix of an initial fusion underperformance issue, gradient-based attribution with statistically validated causal ablation.

**Remaining:**
- React frontend + FastAPI serving layer, built against the now-trained model and validated interpretability outputs
- Cross-cohort external validation (e.g. METABRIC) — identified as a literature gap, addressing it would strengthen generalization claims beyond within-TCGA cross-validation

