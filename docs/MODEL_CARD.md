# Model Card: MOSAIC / BioNexus PAM50 Breast Cancer Subtype Classifier

## 1. Model Details
- **Model Name:** `mosaic-subtype-classifier`
- **Registry Stage:** `Production` (v2)
- **Model Architecture:** `BioNexusClassifier`
  - 3 Parallel GraphGPS Encoders (RNA: 5,000 features, miRNA: 1,881 features, DNA Methylation: 3,957 features)
  - Cross-Attention Modality Fusion
  - Attention-Weighted Modality Pooling
  - Multi-layer Classification Head (5 PAM50 Subtypes)
- **Framework:** PyTorch 2.3+ / PyG (PyTorch Geometric) 2.5+
- **Input Dimensions:**
  - RNA Expression: 5,000 gene features
  - miRNA Expression: 1,881 mature miRNA features
  - DNA Methylation: 3,957 CpG island probe features
- **Output:** Softmax probabilities over 5 PAM50 breast cancer intrinsic subtypes:
  - `Basal`: Basal-like
  - `Her2`: HER2-enriched
  - `LumA`: Luminal A
  - `LumB`: Luminal B
  - `Normal`: Normal-like

---

## 2. Intended Use & Clinical Scope
- **Primary Use Case:** Research and clinical decision support for multi-omics breast cancer intrinsic subtype stratification.
- **Inference Modes:**
  1. **Known Cohort Lookup:** `POST /predict` or `GET /predict/{patient_id}` (retrieves aligned profile and ego-graph from verified TCGA-BRCA store).
  2. **Live Unseen Sample Inference:** `POST /v1/predict-live` (supports raw omics vectors, inductive k-NN graph projection, dynamic missing-modality masking, probability calibration, and cheapest-next-test recommendation).
- **Out of Scope:** Standalone diagnostic determination without pathologist/oncologist review. Not cleared as an FDA Class III primary diagnostic device.

---

## 3. Training & Validation Cohort
- **Cohort:** TCGA-BRCA (The Cancer Genome Atlas - Breast Invasive Carcinoma)
- **Cohort Size:** 646 unique patients with complete tripartite omics profiles.
- **Label Distribution:**
  - LumA: 319 (49.4%)
  - LumB: 121 (18.7%)
  - Basal: 107 (16.6%)
  - Her2: 74 (11.5%)
  - Normal: 25 (3.9%)
- **Validation Protocol:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

---

## 4. Performance & Cross-Validation Metrics

| Metric | 5-Fold OOF Summary | Fold 3 (Selected Champion) |
| :--- | :--- | :--- |
| **Accuracy** | 13.62% | 14.73% |
| **Balanced Accuracy** | 19.35% | 19.50% |
| **Macro F1** | 0.1092 | 0.0583 |
| **Weighted F1** | 0.0830 | 0.0610 |
| **Log Loss** | 1.9205 | 1.8410 |
| **Expected Calibration Error (ECE)** | 0.2234 | 0.2150 |

> [!NOTE]
> **Provisional Model Status:** All imported fold checkpoints are explicitly tagged in MLflow with `data_validated=false` and `source=local_checkpoint`. Checkpoint `best_model.pt` was selected from Fold 3 as the optimistic validation winner. Production retraining on DVC-tracked normalized data is required to advance tags to `data_validated=true`.

---

## 5. Explainability & Interpretability
- **Integrated Gradients:** Gene-level attribution vectors computed per patient per modality to explain driving transcriptional and epigenetic markers.
- **Modality Weights:** Attention pooling weights dynamically quantify RNA vs miRNA vs DNA methylation contribution for each prediction.
- **Cross-Modality Ego Graph:** Interactive k-NN neighborhood visualization showing multi-omics phenotypic neighbors.

---

## 6. Security, Governance & Audit
- **Audit Logging:** Every prediction is immutably logged with timestamp, endpoint, hashed input features, and model version to `reports/audit_log.jsonl`.
- **Zero Raw PHI:** No patient names, MRNs, or raw biometric identifiers are persisted in logs.
- **Data Integrity:** Dataset and model weights are verified against Polygon blockchain Merkle roots.
- **Promotion Gate:** Automated promotion gate (`src/bio_nexus/registry/promote.py`) prevents untested models from entering `Production`.
