# Contributing to MOSAIC

## Setup
```bash
git clone <repo-url> && cd mosaic
uv pip install -e ".[dev]"
pre-commit install
```

## Branching
- `main` — always deployable, protected, merges only via PR + passing CI.
- `dev` — integration branch.
- `feature/<short-desc>` — all work happens here, branched off `dev`.

## Before opening a PR
```bash
pre-commit run --all-files
pytest tests/unit -v
```
CI will re-run all of this anyway, but catching it locally is faster than waiting on GitHub Actions.

## Experiment tracking
Every training run must go through `mosaic.train`, which auto-logs to MLflow.
Don't run ad-hoc training scripts that bypass logging — if it's not in MLflow, it didn't happen, as far as the report is concerned.

## Data
Raw/processed omics data is DVC-tracked, not git-tracked. Run `dvc pull` after clone to fetch data (requires remote access — ask for credentials).
