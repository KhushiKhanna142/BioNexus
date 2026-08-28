# BioNexus – Multi-Omics Graph Transformer for Cancer Subtype Classification

BioNexus is a production-grade MLOps pipeline for multi-omics cancer subtype classification. It integrates gene expression (RNA-seq), DNA methylation, and miRNA expression into a unified biological graph and leverages Graph Transformer architectures for representation learning.

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
├── MOSAIC_Project_Documentation.docx  Full project documentation
│
├── data/
│   ├── sample_sheets/              Organized GDC Sample Sheets
│   │   ├── RNA SAMPLE SHEET/
│   │   ├── MIRNA SAMPLE SHEET/
│   │   └── METHYLATION SAMPLE SHEET/
│   ├── raw/                        Raw omics downloads (rnaseq, mirna, methylation)
│   └── processed/                  Processed matrices & locked-in cohort
│       └── final/                  Final aligned datasets (rna, mirna, meth, labels)
│
├── src/bio_nexus/                  Modular Python Package
│   ├── data/                       Stage 1 — ingestion & preprocessing
│   ├── graph/                      Stage 2 — gene-interaction graph construction
│   ├── encoding/                   Stage 3 — graph transformer encoding
│   ├── fusion/                     Stage 4 — multi-omics cross-attention fusion
│   ├── models/                     Stage 5a — classification head
│   ├── interpretability/           Stage 5b — causal ablation validation
│   ├── evaluation/                 metrics & cross-cohort generalization
│   ├── registry/                   MLflow model lifecycle
│   ├── serving/                    FastAPI inference API
│   ├── utils/                      config, seeding, logging, cloud utilities
│   └── train.py                    training entrypoint
│
├── configs/                        training & experiment configurations
├── tests/                          unit & model quality gate tests
├── docker/                         Dockerfile, Dockerfile.dev, Dockerfile.serving
├── k8s/                            Kubernetes deployment manifests
├── pyproject.toml                  Python build & dependency configuration
└── dvc.yaml                        DVC data pipeline specification
```

---

## Data Pipeline Execution

Run the 5 pipeline steps sequentially from `scripts/`:

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

| Subtype | Description | Patient Count |
| :--- | :--- | :--- |
| **LumA** | Luminal A | 336 |
| **LumB** | Luminal B | 129 |
| **Basal** | Basal-like | 113 |
| **Her2** | HER2-enriched | 39 |
| **Normal** | Normal-like | 29 |
| **Total** | **Multi-Omics Aligned** | **646** |
