"""
Inference API — serves the current Production model from the MLflow
registry (never a locally hardcoded model file; that's how you end up
serving a stale model with no idea which version is live).

Uses FastAPI rather than Flask: async-friendly, typed request/response
models via Pydantic (matches the config pattern used everywhere else in
this repo), and free OpenAPI docs at /docs — useful when a recruiter or
your professor wants to poke at the API directly.

Run locally:
    uvicorn mosaic.serving.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health    liveness/readiness probe for Kubernetes
    GET  /metrics   Prometheus scrape endpoint
    POST /predict   subtype prediction for one patient's multi-omics record
"""
from __future__ import annotations

import os
import time

import mlflow
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from mosaic.registry.register_model import MODEL_NAME
from mosaic.utils.logging import get_logger

logger = get_logger(__name__)
app = FastAPI(title="MOSAIC Inference API", version="0.1.0")

PREDICTION_COUNT = Counter(
    "mosaic_predictions_total", "Total prediction requests", ["status"]
)
PREDICTION_LATENCY = Histogram(
    "mosaic_prediction_latency_seconds", "Prediction request latency"
)

_model = None  # lazy-loaded on first request / startup


class PredictRequest(BaseModel):
    """One patient's aligned multi-omics feature vector.

    Exact shape depends on the ML side's final feature schema — this is
    the contract point where serving meets the model; update it once
    Stage 4 fusion output shape is finalized.
    """

    expression_features: list[float]
    methylation_features: list[float]
    mirna_features: list[float]


class PredictResponse(BaseModel):
    predicted_subtype: str
    confidence: float
    model_version: str


def _load_production_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI is not set.")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}/Production"
    logger.info(f"Loading model from {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


@app.on_event("startup")
def startup() -> None:
    global _model
    try:
        _model = _load_production_model()
    except Exception as e:
        # Don't crash the pod on startup if the registry is briefly
        # unavailable -- fail loudly on /predict instead, so k8s liveness
        # (which hits /health) doesn't loop-crash the deployment.
        logger.error(f"Could not load Production model at startup: {e}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        # Actual feature fusion + inference call goes here once the ML
        # side's fusion/classification modules are implemented.
        raise NotImplementedError("Wire this to mosaic.fusion + mosaic.models.classifier")
    except NotImplementedError:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=501, detail="Model inference not yet implemented")
    finally:
        PREDICTION_LATENCY.observe(time.time() - start)
