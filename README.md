<<<<<<< HEAD
# MOSAIC — Multi-Omics Graph Transformer for Cancer Subtype Classification

Final-year B.Tech (AIML) project, SIES Graduate School of Technology.
Fuses gene expression, DNA methylation, and miRNA expression into a unified
graph and applies graph transformer architectures to classify cancer
subtypes — with causal-ablation interpretability and cross-cohort
generalization testing (see `docs/` for full project documentation).

## Repo layout

```
src/mosaic/
  data/             Stage 1 — ingestion & preprocessing
  graph/            Stage 2 — gene-interaction graph construction
  encoding/         Stage 3 — graph transformer encoding
  fusion/           Stage 4 — multi-omics fusion
  models/           Stage 5a — classification head
  interpretability/ Stage 5b — causal ablation (novelty claim #1)
  evaluation/       metrics, cross-cohort accuracy drop (novelty claim #2)
  registry/         MLflow model lifecycle: register -> Staging -> Production
  serving/          FastAPI inference API (loads Production model, exposes
                     /predict, /health, /metrics)
  utils/            config, seeding, logging, S3/cloud connector
  train.py          single entrypoint that wires all stages together

configs/            all hyperparameters, no hardcoding in code
tests/
  unit/             fast, no external services
  model/            quality gate — must pass before promote_model.py runs
docker/
  Dockerfile          training image
  Dockerfile.dev       local dev / jupyter
  Dockerfile.serving   lean inference-only image
k8s/                Deployment + Service manifests for the inference API
.github/workflows/
  ci.yml            lint, type-check, unit tests, smoke-test, docker build
  cd.yml            on version tag: quality gate -> promote model ->
                     build+push training & serving images -> (k8s deploy,
                     disabled until you have a cluster)
dvc.yaml            data pipeline — raw TCGA data is DVC-tracked, not git
```

## Model lifecycle

Training run -> MLflow logs metrics/artifacts -> `mosaic.registry.register_model`
registers it as **Staging** -> `tests/model/` quality gate must pass ->
`mosaic.registry.promote_model` archives the old Production version and
promotes Staging -> Production -> the serving API always loads
`models:/mosaic-subtype-classifier/Production`, so a bad model never
reaches the API without first passing the gate.

## Required secrets (GitHub Actions)

| Secret | Used for |
|---|---|
| `MLFLOW_TRACKING_URI` | model registry + experiment tracking (self-hosted or [DagsHub](https://dagshub.com) free tier — good option for a student project since it's hosted and shareable) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | DVC remote + raw data pulls, if using S3 |
| `KUBE_CONFIG` | only needed once you enable the k8s deploy job |

## Ownership

MLOps (infra, CI/CD, Docker, configs, experiment tracking, reproducibility)
is owned end-to-end by the MLOps lead. Model/algorithm code in `data/`,
`graph/`, `encoding/`, `fusion/`, `interpretability/` is owned by the ML
lead. See `CODEOWNERS`.

## Quickstart

```bash
# local dev
uv pip install -e ".[dev]"
pre-commit install

# run tests
pytest tests/unit -v

# local training stack (MLflow + training container)
docker compose up

# production image
docker build -f docker/Dockerfile -t mosaic:latest .
```

## Experiment tracking

All training runs log to MLflow (self-hosted via `docker-compose.yml`,
`http://localhost:5000`). Every run records config, metrics (accuracy,
macro-F1, AUC-ROC, ablation importance score, cross-cohort accuracy drop),
and artifacts. No results should be reported that aren't backed by a
logged MLflow run.

## Data

Raw and processed omics data are DVC-tracked (`dvc.yaml`), not committed
to git — genomics data is large and shouldn't live in version control
history. Run `dvc pull` after cloning (requires DVC remote access).

## Status

Infra scaffold complete. Pipeline stages (`NotImplementedError` stubs in
`src/mosaic/`) are pending ML implementation — see `CONTRIBUTING.md` for
the branching/PR workflow.
=======
# bio_nexus
>>>>>>> 3d187f6d5aa79c7de5e48503878965a13e0ea84b
