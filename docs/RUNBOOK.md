# Production Operations Runbook: MOSAIC / BioNexus

This runbook guides operators, MLOps engineers, and reviewers in managing, monitoring, troubleshooting, and deploying the MOSAIC cancer subtype classification stack.

---

## 1. Architecture & Port Reference

| Service | Host Port | Container Port | Protocol / Path | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **API (FastAPI)** | `5000` | `8000` | HTTP (`/health`, `/predict`) | Primary REST inference and artifact serving |
| **MLflow Server** | `5001` | `5000` | HTTP (`/health`, UI) | MLflow tracking & model registry store |
| **Web Dashboard** | `5173` | `5173` | HTTP (Vite React) | Interactive clinician dashboard |

---

## 2. Quickstart & Deployment

### Start Stack via Docker Compose
```bash
docker compose up -d --build
```

### Verify Service Health
```bash
# Check FastAPI serving status and model readiness
curl http://localhost:5000/health

# Check MLflow tracking server health
curl http://localhost:5001/health
```

Expected health response from API:
```json
{
  "status": "healthy",
  "mode": "live",
  "model_loaded": true,
  "model_source": "mlflow_registry",
  "model_version": "models:/mosaic-subtype-classifier/Production"
}
```

---

## 3. Standard Operating Procedures (SOP)

### A. Making a Subtype Prediction (Known Cohort Patient)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "TCGA-3C-AAAU"}'
```

### B. Making a Live Unseen Prediction (Raw Omics)
```bash
curl -X POST http://localhost:5000/v1/predict-live \
  -H "Content-Type: application/json" \
  -d '{
    "raw_features": {
      "rna": {"TP53": 12.4, "BRCA1": 8.1, "ESR1": 15.2},
      "mirna": {"hsa-let-7a-1": 6.8},
      "meth": {"cg00000292": 0.85}
    }
  }'
```

### C. Running Automated Unit, Integration & Model Tests
```bash
# Run model quality gates and registry loading tests
pytest tests/model -v

# Run full serving API test suite
pytest tests/api tests/unit -v
```

### D. Automated Model Promotion Gate (Phase 6)
To promote a new candidate model from `Staging` to `Production`:
```bash
python -m bio_nexus.registry.promote --model-name mosaic-subtype-classifier
```
This script evaluates contract validity, output shapes, and metrics against baseline thresholds before moving the model to `Production`.

### E. Drift Monitoring & Retraining Assessment (Phase 8)
```bash
python -m bio_nexus.monitoring.drift
```
Evaluates KS-test distribution drift on omics features and Chi-square goodness of fit on predicted subtypes. If drift exceeds 20%, an automated retraining recommendation is flagged to `reports/drift_report.json`.

---

## 4. Incident Response & Troubleshooting

### Scenario 1: API returns `503 Service Unavailable` on `/health` or `/predict`
**Symptoms:** `/health` status is `degraded` and `model_loaded: false`.
**Root Cause:** The champion model artifact failed to load from MLflow or MLflow server is unreachable.
**Remediation Steps:**
1. Check MLflow container status:
   ```bash
   docker compose ps mlflow
   docker compose logs mlflow
   ```
2. Verify model registration in `mlflow.db`:
   ```bash
   python scripts/check_mlflow_model.py
   ```
3. Restart API container with logs:
   ```bash
   docker compose restart api
   docker compose logs -f api
   ```

### Scenario 2: Model artifacts path mismatch between Host and Docker
**Symptoms:** Error `No such file or directory: 'C:/...'` inside Docker logs.
**Resolution:** The stack includes automatic path translation in `artifacts.py` and `ml_client.py` that remaps host paths to `/mlruns` container mounts. Ensure `./mlruns:/mlruns:ro` is present in `docker-compose.yml`.

### Scenario 3: Memory or CUDA latency issues during local evaluation
**Resolution:** Set environment variable `CUDA_VISIBLE_DEVICES=""` to force CPU evaluation and eliminate Anaconda NVML probe latency.
