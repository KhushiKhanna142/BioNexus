# MOSAIC / BioNexus Repository Cleanup Report (Phase 1: Audit)

**Date**: October 20, 2026  
**Repository**: `bio_nexus` (MOSAIC Multi-Omics Graph Transformer for PAM50 Classification)  
**Branch**: `chore/repo-cleanup`  
**Checkpoint Commit**: `dec811b` (preserves complete pre-cleanup state)  
**Status**: Read-Only Audit Complete — Awaiting Approval for Phase 2  

---

## Executive Summary

This audit assesses the state of `bio_nexus` to establish an intuitive, industry-grade repository structure. The primary objective is to enable any machine learning engineer or industry reviewer to understand, navigate, and verify the system within five minutes, while ensuring absolute preservation of trained model checkpoints, cohort splits, evaluation results, and pipeline stability.

### Key Audit Findings
1. **Repository Footprint**: Total non-ignored working tree size is **~305 MB** (excluding `.git` and `node_modules`).
2. **Tracked Large Files**: `data/serving/*.parquet` matrices (~82 MB total) and `MOSAIC_Project_Documentation.docx` (31 KB) are currently tracked in git. Three loose ChatGPT image files (~5 MB total) sit in the root (currently gitignored via `*.png`).
3. **Redundant & Deprecated Components**:
   - `frontend_old/` (77.5 KB) and `frontend/` (346.7 KB) are legacy prototypes completely superseded by `frontend-new/` (15.1 MB).
   - Scratch and one-off debug scripts (`scratch_*.py`, `check_cohort.py`, `scripts/step1_*.py`, `scripts/test_*.py`) clutter the root and scripts directories.
4. **Namespace Inconsistency**: `src/model_comparison/` sits outside the main package namespace (`src/bio_nexus/`).
5. **Infrastructure & Documentation Mismatches**:
   - `docker-compose.yml` specifies a build context for `web` targeting `./frontend-new/Dockerfile`, but `frontend-new/Dockerfile` does not exist.
   - `README.md` lists legacy `/api/*` endpoints and states the API runs on port 8000, whereas Docker Compose exposes it on port 5000.
   - Two overlapping promotion scripts exist: `src/bio_nexus/registry/promote_model.py` and `src/bio_nexus/registry/promote.py`.

---

## a) Current Directory Tree (3 Levels Deep with Sizes)

```text
bio_nexus/ (305.4 MB)
├── .github/ (4.9 KB)
│   ├── ISSUE_TEMPLATE/ (183 B)
│   │   └── bug_report.md (183 B)
│   ├── workflows/ (4.3 KB)
│   │   ├── cd.yml (2.1 KB)
│   │   └── ci.yml (2.2 KB)
│   └── pull_request_template.md (478 B)
├── configs/ (2.8 KB)
│   ├── training/ (1.1 KB)
│   │   ├── default.yaml (456 B)
│   │   └── smoke_test.yaml (647 B)
│   ├── gateway_config.yaml (711 B)
│   └── qc.yaml (1.1 KB)
├── contracts/ (8.8 KB)
│   ├── PatientRecordRegistry.json (6.4 KB)
│   └── PatientRecordRegistry.sol (2.5 KB)
├── data/ (141.5 MB)
│   ├── audit/ (6.1 KB)
│   │   └── audit.log (6.1 KB)
│   ├── cases/ (1.8 MB) [534 patient case subdirectories]
│   ├── processed/ (53.7 MB) [PROTECTED: DVC / Trained Checkpoints]
│   │   ├── checkpoints/ (26.4 MB) [best_model.pt, fold checkpoints]
│   │   ├── final/ (13.4 KB)
│   │   ├── graphs/ (27.4 MB)
│   │   ├── results/ (963 B)
│   │   └── .gitkeep (0 B)
│   ├── raw/ (0 B)
│   │   └── .gitkeep (0 B)
│   ├── sample_sheets/ (1.1 MB)
│   │   ├── METHYLATION SAMPLE SHEET/ (274.3 KB)
│   │   ├── MIRNA SAMPLE SHEET/ (567.4 KB)
│   │   └── RNA SAMPLE SHEET/ (289.5 KB)
│   ├── security/ (261.7 KB)
│   │   ├── records/ (1.9 KB)
│   │   ├── cohort_merkle_batch.json (259.8 KB)
│   │   ├── vault_master.key (64 B) [git-ignored]
│   │   └── vault_salt.txt (32 B) [git-ignored]
│   ├── serving/ (83.8 MB)
│   │   ├── graphs/ (522.4 KB)
│   │   ├── *.parquet matrices (77.8 MB)
│   │   └── *.json metadata (causal_validation, manifest, labels, etc.)
│   ├── brca_tcga_pan_can_atlas_2018_clinical_data.tsv (573.7 KB)
│   └── splits.json (136.6 KB) [PROTECTED]
├── docker/ (2.2 KB)
│   ├── Dockerfile (778 B)
│   ├── Dockerfile.dev (343 B)
│   └── Dockerfile.serving (1.1 KB)
├── docs/ (18.1 KB)
│   ├── sample_reports/ (12.1 KB)
│   └── API.md (6.1 KB)
├── frontend/ (346.7 KB) [LEGACY PROTOTYPE]
│   ├── dist/ (197.6 KB)
│   ├── src/ (74.4 KB)
│   └── package.json (427 B)
├── frontend-new/ (15.1 MB) [ACTIVE REACT APPLICATION]
│   ├── public/ (6.6 MB)
│   ├── src/ (558.9 KB)
│   │   ├── api/, components/, data/, lib/, pages/
│   │   └── App.tsx, index.css, main.tsx
│   ├── package.json (1.2 KB)
│   └── vite.config.ts (899 B)
├── frontend_old/ (77.5 KB) [DEPRECATED PROTOTYPE]
│   ├── src/ (19.3 KB)
│   └── package.json (373 B)
├── k8s/ (1.5 KB)
│   ├── deployment.yaml (1.3 KB)
│   └── service.yaml (238 B)
├── mlruns/ (144.1 MB) [LOCAL MLFLOW STORE - PROTECTED]
│   ├── 1/, 2/, 3/ [experiments, runs, artifacts]
│   └── mlflow.db (1.4 MB)
├── models/ (0 B)
│   └── .gitkeep (0 B)
├── reports/ (235.7 KB)
│   ├── prompts/ (1.5 KB)
│   ├── checkpoint_evaluation.json (2.3 KB)
│   ├── drift_report.json (1.2 KB)
│   └── experiment_info.json (371 B)
├── results/ (6.3 KB)
│   └── f3/ (6.3 KB) [gain_table.json, modality_subsets.json, policy_curve.csv]
├── scripts/ (205.6 KB) [34 scripts]
│   ├── import_checkpoint.py (12.6 KB)
│   ├── evaluate_checkpoints.py (9.0 KB)
│   ├── patient_cohort.py (4.4 KB)
│   ├── preprocess.py (12.1 KB)
│   ├── run_serving_tests.py (5.6 KB)
│   └── [scratch & redundant scripts]
├── src/ (432.2 KB)
│   ├── bio_nexus/ (367.8 KB)
│   │   ├── data/, encoding/, evaluation/, fusion/, graph/
│   │   ├── interpretability/, models/, monitoring/, observability/
│   │   ├── registry/, security/, serving/, utils/
│   │   └── train.py (17.4 KB)
│   └── model_comparison/ (64.4 KB) [UNNAMESPACED MODULE]
│       ├── benchmark/, data/, tests/
│       └── analysis.py, data_source.py, registry.py, router.py
├── tests/ (49.8 KB)
│   ├── api/ (17.8 KB)
│   ├── integration/ (0 B)
│   ├── model/ (5.1 KB)
│   ├── unit/ (1.4 KB)
│   └── [root tests: test_batch_triage, test_cost_accuracy, test_new_case_qc, test_report_pdf]
├── [Root loose files]
│   ├── .dockerignore (142 B), .env (275 B), .env.example (680 B), .gitignore (912 B)
│   ├── BioNexus_Project_Writeup.md (53.1 KB)
│   ├── bionexus_spec.json (432 B)
│   ├── ChatGPT Image *.png (3 files, ~4.9 MB total)
│   ├── check_cohort.py (456 B)
│   ├── CODEOWNERS (616 B), CONTRIBUTING.md (935 B), LICENSE (12.2 KB)
│   ├── docker-compose.yml (1.7 KB), dvc.yaml (1.6 KB), Makefile (844 B), params.yaml (899 B)
│   ├── metrics_seed.py (2.8 KB), mlflow.db (1.4 MB)
│   ├── MODEL_CARD.md (3.7 KB), RUNBOOK.md (3.8 KB)
│   ├── MOSAIC_Project_Documentation.docx (31.1 KB)
│   ├── prompt_9_full.txt (11.1 KB), pyproject.toml (1.4 KB)
│   ├── railway.json (281 B), render.yaml (442 B), vercel.json (344 B)
│   ├── requirements-dev.lock (566 B), requirements-serve.lock (411 B)
│   └── scratch_test_import.py (326 B)
```

---

## b) Candidates for Removal, Move, or Git-Ignore

| Path | Why it Exists | Who References It | Size | Recommendation | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `prompt_9_full.txt` | Raw prompt dump from earlier development | None | 11.1 KB | **delete** | High |
| `ChatGPT Image Sep 30, 2026, 11_34_36 PM.png` | Ideation diagram screenshot | None (git-ignored) | 1.7 MB | **delete** (or move to `docs/assets/`) | High |
| `ChatGPT Image Sep 30, 2026, 11_35_38 PM.png` | Ideation diagram screenshot | None (git-ignored) | 1.6 MB | **delete** (or move to `docs/assets/`) | High |
| `ChatGPT Image Sep 30, 2026, 11_36_30 PM.png` | Ideation diagram screenshot | None (git-ignored) | 1.6 MB | **delete** (or move to `docs/assets/`) | High |
| `MOSAIC_Project_Documentation.docx` | Binary Word document of documentation | None (tracked in git) | 31.1 KB | **move** to `docs/archive/` or **delete** | High |
| `BioNexus_Project_Writeup.md` | Comprehensive ML/biology methodology paper | README.md | 53.1 KB | **move** to `docs/PROJECT_WRITEUP.md` | High |
| `MODEL_CARD.md` | Formal Model Card for `mosaic-subtype-classifier` | RUNBOOK.md, schemas.py | 3.7 KB | **move** to `docs/MODEL_CARD.md` | High |
| `RUNBOOK.md` | Operations runbook & SOPs | README.md | 3.8 KB | **move** to `docs/RUNBOOK.md` | High |
| `bionexus_spec.json` | Feature tensor dimension specifications | adapter.py, benchmark | 432 B | **move** to `configs/bionexus_spec.json` | High |
| `check_cohort.py` | One-off cohort validation test | None | 456 B | **delete** | High |
| `scratch_test_import.py` | One-off test importing BioNexusModelAdapter | None | 326 B | **delete** | High |
| `metrics_seed.py` | Standalone script generating sample metrics | README.md | 2.8 KB | **move** to `scripts/metrics_seed.py` | High |
| `mlflow.db` (root) | Local SQLite MLflow tracking DB | ml_client.py fallback | 1.4 MB | **git-ignore & keep local** (never commit) | High |
| `mlruns/mlflow.db` | Bind-mounted duplicate for Docker container | docker-compose.yml | 1.4 MB | **keep** (required for container mount) | High |
| `frontend_old/` | Explicitly deprecated older frontend prototype | .gitignore only | 77.5 KB | **delete** | High |
| `frontend/` | Older intermediate frontend prototype | .gitignore only | 346.7 KB | **delete** (superseded by frontend-new) | Medium |
| `src/model_comparison/` | Benchmark and model registry router | main AppLayout, frontend | 64.4 KB | **move** to `src/bio_nexus/model_comparison/` | High |
| `tests/test_batch_triage.py` | Root-level test file | pytest runner | 5.8 KB | **move** to `tests/api/test_batch_triage.py` | High |
| `tests/test_cost_accuracy.py` | Root-level test file | pytest runner | 5.7 KB | **move** to `tests/api/test_cost_accuracy.py` | High |
| `tests/test_new_case_qc.py` | Root-level test file | pytest runner | 8.6 KB | **move** to `tests/api/test_new_case_qc.py` | High |
| `tests/test_report_pdf.py` | Root-level test file | pytest runner | 5.4 KB | **move** to `tests/api/test_report_pdf.py` | High |
| `scripts/check_deps.py` | Temporary dependency checker | None | 329 B | **delete** | High |
| `scripts/check_oof.py` | Diagnostic script for out-of-fold data | None | 266 B | **delete** (or merge into tests) | High |
| `scripts/scratch_inventory.py`| Scratch inventory script | None | 3.7 KB | **delete** | High |
| `scripts/step1_cohort.py` | Redundant Step 1 cohort inspection | None | 1.5 KB | **delete** (superseded by patient_cohort.py) | High |
| `scripts/step1_inspect.py` | Redundant Step 1 checkpoint inspect | None | 878 B | **delete** (superseded by inspect_checkpoints.py) | High |
| `scripts/test_api_dash.py` | Ad-hoc endpoint test | None | 512 B | **delete** (covered by pytest tests/api) | High |
| `scripts/test_xgb.py` | Ad-hoc XGBoost test | None | 390 B | **delete** (covered by benchmark tests) | High |
| `scripts/verify_dashboard.py`| Ad-hoc dashboard connectivity check | None | 797 B | **delete** (covered by test_endpoints.py) | High |
| `src/bio_nexus/registry/promote_model.py` | Legacy promotion script without quality gates | cd.yml | 3.3 KB | **consolidate** with `promote.py` | High |
| `data/splits.json` | 5-fold cross-validation patient split mapping | train.py, adapter | 136.6 KB | **KEEP** (protected by Rule 3) | High |
| `data/processed/checkpoints/` | Trained PyTorch model checkpoints | MLflow, evaluation | 26.4 MB | **KEEP** (protected by Rule 3) | High |
| `.env` | Local environment variables & secrets | Docker, local env | 275 B | **KEEP & ensure git-ignored** (never print) | High |

---

## c) Documentation vs. Reality Mismatches

| Source File | Documented Item | Reality / Actual System State | Severity |
| :--- | :--- | :--- | :--- |
| `docker-compose.yml` (line 57) | `web` service builds from `./frontend-new/Dockerfile` | `./frontend-new/Dockerfile` **does not exist**; container build fails if `web` target is requested | **High** |
| `README.md` (lines 130–134) | Documented endpoints as `/api/health`, `/api/patients`, `/api/predict/{id}`, `/api/metrics` | Actual FastAPI routes in `api.py` are mounted at root: `/health`, `/patients`, `/predict`, `/predict/{id}`, `/metrics`, `/v1/predict-live` | **High** |
| `README.md` (line 124) | States API inference runs on `http://localhost:8000` | In Docker Compose (`docker-compose.yml`), API port is mapped to **`5000:8000`** (`http://localhost:5000`) | **Medium** |
| `README.md` (lines 34–78) | Tree diagram shows loose `BioNexus_Project_Writeup.md` in root and lacks MLOps documentation | Does not mention `MODEL_CARD.md`, `RUNBOOK.md`, `CLEANUP_REPORT.md`, or `reports/` | **Medium** |
| `.github/workflows/cd.yml` | Calls `python -m bio_nexus.registry.promote_model` | Active quality gate logic is in `bio_nexus.registry.promote` (which evaluates accuracy > 0.85 and (N, 5) shapes) | **Medium** |
| `Makefile` (line 17) | `pip install -e .[dev]` | Lockfiles exist (`requirements-dev.lock`, `requirements-serve.lock`), but Makefile does not reference locked installation | **Low** |
| `src/model_comparison/` | Located directly under `src/` | Violates package rule: all Python library code belongs inside `src/bio_nexus/` | **Medium** |

---

## d) Proposed Target Layout & Move Map

### Target Layout
```text
bio_nexus/
├── README.md                      # High-level overview, architecture diagram, 5-minute quickstart
├── LICENSE                        # Apache 2.0 / MIT license
├── CHANGELOG.md                   # Version history and Phase 2 removals list
├── pyproject.toml                 # Package definition and entrypoints
├── Makefile                       # Developer automation: setup, lint, test, up, down, repro
├── dvc.yaml                       # Data & pipeline stage orchestration
├── params.yaml                    # Hyperparameters and pipeline configuration
├── requirements-dev.lock          # Reproducible pinned dev dependencies
├── requirements-serve.lock        # Reproducible pinned serving dependencies
├── .env.example                   # Sanitized environment template (no real secrets)
├── .gitignore                     # Comprehensive ignore rules
├── .dockerignore                  # Docker build context exclusion rules
├── .pre-commit-config.yaml        # Pre-commit hooks: ruff, gitleaks, nbstripout
│
├── configs/                       # Centralized configuration files
│   ├── training/
│   │   ├── default.yaml
│   │   └── smoke_test.yaml
│   ├── bionexus_spec.json         # (moved from root)
│   ├── gateway_config.yaml
│   └── qc.yaml
│
├── docs/                          # All documentation in one discoverable location
│   ├── MODEL_CARD.md              # (moved from root)
│   ├── RUNBOOK.md                 # (moved from root)
│   ├── PROJECT_WRITEUP.md         # (moved from root BioNexus_Project_Writeup.md)
│   ├── CLEANUP_REPORT.md          # This audit report
│   ├── ARCHITECTURE.md            # System & pipeline architecture reference
│   ├── API.md                     # Endpoint documentation
│   └── sample_reports/            # Clinical PDF report examples
│
├── src/bio_nexus/                 # Single installable package namespace
│   ├── data/                      # Data loaders and ingestion validation
│   ├── encoding/                  # GraphGPS positional and node encodings
│   ├── evaluation/                # Evaluation metrics and cross-cohort tests
│   ├── fusion/                    # Cross-attention multimodal fusion module
│   ├── graph/                     # kNN graph construction
│   ├── interpretability/          # Integrated Gradients & statistical validation
│   ├── model_comparison/          # (moved from src/model_comparison/)
│   ├── models/                    # Classifier head and BioNexusModelAdapter
│   ├── monitoring/                # KS-test & Chi-squared drift detection
│   ├── observability/             # Structured logging and Prometheus metrics
│   ├── registry/                  # MLflow registration and quality-gated promotion
│   ├── security/                  # Cryptographic hashing & audit ledger
│   ├── serving/                   # Production FastAPI inference service
│   ├── utils/                     # Configs, seeding, logging utilities
│   ├── __init__.py
│   └── train.py                   # 5-fold CV training entry point
│
├── scripts/                       # Clean CLI entry points (documented & listed in README)
│   ├── __init__.py
│   ├── import_checkpoint.py       # MLflow model import with dynamic dimensions
│   ├── evaluate_checkpoints.py    # Held-out fold checkpoint evaluation
│   ├── metrics_seed.py            # (moved from root)
│   ├── patient_cohort.py          # 646 patient cohort extraction
│   ├── preprocess.py              # Omics matrix normalization
│   ├── run_serving_tests.py       # Automated endpoint test suite
│   ├── fix_alembic.py             # Alembic schema version sync utility
│   └── deploy_polygon.py          # Audit ledger contract deployment
│
├── tests/                         # Full test suite structured to mirror src/
│   ├── api/                       # API and endpoint tests
│   │   ├── test_serving.py
│   │   ├── test_v1_product_api.py
│   │   ├── test_batch_triage.py   # (moved from tests/ root)
│   │   ├── test_cost_accuracy.py  # (moved from tests/ root)
│   │   ├── test_new_case_qc.py    # (moved from tests/ root)
│   │   └── test_report_pdf.py     # (moved from tests/ root)
│   ├── model/                     # Model adapter & registry contract tests
│   │   ├── test_adapter.py
│   │   └── test_registered_model.py
│   ├── unit/                      # Graph, config, and ingestion unit tests
│   └── integration/               # End-to-end integration tests
│
├── docker/                        # All container definitions and compose stacks
│   ├── Dockerfile                 # Production training image
│   ├── Dockerfile.dev             # Local development & testing image
│   ├── Dockerfile.serving         # Hardened non-root FastAPI serving image
│   └── Dockerfile.frontend        # Frontend static build & Nginx container
│
├── frontend-new/                  # Active modern React + Vite dashboard
│   ├── src/, public/, package.json, vite.config.ts
│
├── data/                          # Protected: DVC-tracked & local serving assets
├── mlruns/                        # Protected: MLflow experiment tracking & registry
└── reports/                       # Generated audit, evaluation, and drift reports
```

### Proposed `git mv` Move Map

| # | Current Path | Proposed New Path | Action | Rationale |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `BioNexus_Project_Writeup.md` | `docs/PROJECT_WRITEUP.md` | `git mv` | Consolidate documentation into `docs/` |
| 2 | `MODEL_CARD.md` | `docs/MODEL_CARD.md` | `git mv` | Place formal Model Card in `docs/` |
| 3 | `RUNBOOK.md` | `docs/RUNBOOK.md` | `git mv` | Place operational runbook in `docs/` |
| 4 | `bionexus_spec.json` | `configs/bionexus_spec.json` | `git mv` | Centralize all configuration/spec files in `configs/` |
| 5 | `metrics_seed.py` | `scripts/metrics_seed.py` | `git mv` | Clear loose scripts from repo root |
| 6 | `src/model_comparison/` | `src/bio_nexus/model_comparison/` | `git mv` | Encapsulate all Python library code within `bio_nexus` |
| 7 | `tests/test_batch_triage.py` | `tests/api/test_batch_triage.py` | `git mv` | Move root test to `tests/api/` |
| 8 | `tests/test_cost_accuracy.py` | `tests/api/test_cost_accuracy.py` | `git mv` | Move root test to `tests/api/` |
| 9 | `tests/test_new_case_qc.py` | `tests/api/test_new_case_qc.py` | `git mv` | Move root test to `tests/api/` |
| 10 | `tests/test_report_pdf.py` | `tests/api/test_report_pdf.py` | `git mv` | Move root test to `tests/api/` |
| 11 | `MOSAIC_Project_Documentation.docx` | `docs/archive/MOSAIC_Project_Documentation.docx` | `git mv` | Remove loose binary file from repo root |

---

## Items Recommended for Deletion (Awaiting Your Explicit Approval)

| # | Item Path | Item Type | Size | Justification |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `prompt_9_full.txt` | File | 11.1 KB | Raw LLM prompt artifact from prior interaction; unused by code or tests |
| 2 | `check_cohort.py` | File | 456 B | One-off check script in root; superseded by `scripts/patient_cohort.py` |
| 3 | `scratch_test_import.py` | File | 326 B | One-off verification script; full test suite in `tests/model/test_adapter.py` |
| 4 | `frontend_old/` | Directory | 77.5 KB | Explicitly deprecated legacy frontend prototype |
| 5 | `frontend/` | Directory | 346.7 KB | Superseded early prototype; completely replaced by `frontend-new/` |
| 6 | `scripts/scratch_inventory.py` | File | 3.7 KB | One-off scratch file |
| 7 | `scripts/step1_cohort.py` | File | 1.5 KB | Redundant step 1 script; superseded by `scripts/patient_cohort.py` |
| 8 | `scripts/step1_inspect.py` | File | 878 B | Redundant step 1 script; superseded by `scripts/inspect_checkpoints.py` |
| 9 | `scripts/check_deps.py` | File | 329 B | One-off temporary dependency test |
| 10 | `scripts/test_api_dash.py` | File | 512 B | Ad-hoc test; fully covered by `tests/api/test_serving.py` |
| 11 | `scripts/test_xgb.py` | File | 390 B | Ad-hoc baseline check; covered by unit/benchmark tests |
| 12 | `scripts/verify_dashboard.py` | File | 797 B | Ad-hoc check; covered by `tests/api/` |
| 13 | `ChatGPT Image Sep 30, 2026, 11_34_36 PM.png` | File | 1.7 MB | Loose ideation screenshot in root (can be deleted or moved to `docs/assets/`) |
| 14 | `ChatGPT Image Sep 30, 2026, 11_35_38 PM.png` | File | 1.6 MB | Loose ideation screenshot in root (can be deleted or moved to `docs/assets/`) |
| 15 | `ChatGPT Image Sep 30, 2026, 11_36_30 PM.png` | File | 1.6 MB | Loose ideation screenshot in root (can be deleted or moved to `docs/assets/`) |

---

## Phase 1 Audit Conclusion & Next Steps

Phase 1 audit is complete and read-only. No files have been deleted, moved, or altered.

Per Safety Rule 7: **Execution is paused until you review this report and give approval.**

Please review the proposed move map and candidate deletion list. Let me know:
1. If you approve the deletions item-by-item (or approve all listed candidates).
2. If you would prefer the ChatGPT architecture images deleted or moved to `docs/assets/`.
3. If you are ready for Phase 2 (Restructure).
