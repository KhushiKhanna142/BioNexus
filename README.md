# BioNexus – Multi-Omics Graph Transformer for Cancer Subtype Classification

Final-year B.Tech (AIML) project, SIES Graduate School of Technology.

BioNexus is a production-grade MLOps pipeline for multi-omics cancer subtype classification. It integrates gene expression, DNA methylation, and miRNA expression into a unified biological graph and leverages Graph Transformer architectures for representation learning. The project emphasizes reproducible ML, experiment tracking, model versioning, deployment automation, causal-ablation interpretability, and cross-cohort generalization.

---

## Repository Layout

```
src/bio_nexus/
  data/             Stage 1 — ingestion & preprocessing
  graph/            Stage 2 — gene-interaction graph construction
  encoding/         Stage 3 — graph transformer encoding
  fusion/           Stage 4 — multi-omics fusion
  models/           Stage 5a — classification head
  interpretability/ Stage 5b — causal ablation (novelty claim #1)
  evaluation/       metrics, cross-cohort accuracy drop (novelty claim #2)
  registry/         MLflow model lifecycle: register → Staging → Production
  serving/          FastAPI inference API (/predict, /health, /metrics)
  utils/            config, seeding, logging, cloud utilities
  train.py          single pipeline entrypoint

configs/            training & experiment configurations
tests/
  unit/             fast unit tests
  model/            quality gate before production promotion
docker/
  Dockerfile          production training image
  Dockerfile.dev      development image
  Dockerfile.serving  inference image
k8s/                Kubernetes deployment manifests
.github/workflows/
  ci.yml            lint, type-check, tests, smoke-test, Docker build
  cd.yml            model promotion & deployment pipeline
dvc.yaml            data pipeline definition
```

---

## Model Lifecycle

Training Run

↓

MLflow Experiment Tracking

↓

Register Model (Staging)

↓

Quality Gate Tests

↓

Promote to Production

↓

FastAPI loads

`models:/bio-nexus-subtype-classifier/Production`

Only models that successfully pass automated quality checks are promoted to Production.

---

## Required GitHub Secrets

| Secret | Purpose |
|---------|---------|
| `MLFLOW_TRACKING_URI` | MLflow experiment tracking & model registry |
| `AWS_ACCESS_KEY_ID` | DVC remote storage |
| `AWS_SECRET_ACCESS_KEY` | DVC remote storage |
| `KUBE_CONFIG` | Kubernetes deployment (optional) |

---

## Ownership

- **MLOps Lead** — Infrastructure, CI/CD, Docker, DVC, MLflow, deployment, reproducibility.
- **ML Lead** — Data processing, graph construction, Graph Transformer, fusion, interpretability, evaluation.

---

## Quickstart

```bash
uv pip install -e ".[dev]"
pre-commit install

pytest tests/unit -v

docker compose up

docker build -f docker/Dockerfile -t bio-nexus:latest .
```

---

## Experiment Tracking

Every training run logs to MLflow, recording:

- Hyperparameters
- Accuracy
- Macro-F1
- ROC-AUC
- Causal-ablation importance
- Cross-cohort accuracy drop
- Model artifacts

---

## Data

Raw and processed multi-omics datasets are tracked with **DVC**, not Git. After cloning:

```bash
dvc pull
```

---

## Status

Infrastructure scaffold complete.

Pipeline components inside `src/bio_nexus/` are currently under implementation.
