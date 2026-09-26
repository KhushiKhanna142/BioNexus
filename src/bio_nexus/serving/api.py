"""
bio_nexus.serving.api
--------------------------
FastAPI inference serving layer for the trained BioNexusClassifier.
Loads the model once at startup, then serves prediction, interpretability,
and performance-metrics endpoints for the React frontend.

Run with:
    uvicorn bio_nexus.serving.api:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bio_nexus.interpretability.attribution import (
    integrated_gradients,
    load_feature_names,
    top_attributed_features,
)
from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.train import load_data

BASE_DIR = Path(__file__).resolve().parents[3]

STATE = {}  # populated at startup: model, graphs, y, node_order, feature_names, metrics


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and data...")
    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(BASE_DIR)
    model = BioNexusClassifier(
        rna_in_features=rna_graph.x.shape[1],
        mirna_in_features=mirna_graph.x.shape[1],
        meth_in_features=meth_graph.x.shape[1],
    )
    ckpt_path = BASE_DIR / "data" / "processed" / "checkpoints" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    STATE["model"] = model
    STATE["graphs"] = [rna_graph, mirna_graph, meth_graph]
    STATE["y"] = y
    STATE["node_order"] = node_order
    STATE["patient_to_idx"] = {pid: i for i, pid in enumerate(node_order)}
    STATE["feature_names"] = {
        "rna": load_feature_names(BASE_DIR, "rna"),
        "mirna": load_feature_names(BASE_DIR, "mirna"),
        "meth": load_feature_names(BASE_DIR, "meth"),
    }

    metrics_path = BASE_DIR / "data" / "processed" / "results" / "metrics.json"
    if metrics_path.exists():
        import json

        STATE["metrics"] = json.loads(metrics_path.read_text())
    else:
        STATE["metrics"] = None
        print(
            f"[WARNING] {metrics_path} not found - /api/metrics will return null. "
            f"See metrics_seed.py to generate it."
        )

    print(f"Loaded model and {len(node_order)} patients. Ready.")
    yield
    STATE.clear()


app = FastAPI(title="BioNexus API", lifespan=lifespan)

# CORS: allow the React dev server (Vite default port 5173, CRA default 3000) to call this API.
# TIGHTEN THIS before deploying publicly - "*" origins is fine for local dev only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    patient_id: str
    true_subtype: str
    predicted_subtype: str
    confidence: float
    all_probabilities: dict
    modality_contribution: dict


class GeneAttribution(BaseModel):
    gene: str
    score: float


class ExplainResponse(BaseModel):
    patient_id: str
    layer: str
    top_genes: list[GeneAttribution]


@app.get("/api/health")
def health():
    return {"status": "ok", "num_patients": len(STATE.get("node_order", []))}


@app.get("/api/patients")
def list_patients():
    """Returns all patient IDs + their true subtype, for the frontend's patient selector."""
    y = STATE["y"]
    return [
        {"patient_id": pid, "true_subtype": PAM50_SUBTYPES[y[i]]}
        for i, pid in enumerate(STATE["node_order"])
    ]


@app.get("/api/predict/{patient_id}", response_model=PredictionResponse)
def predict(patient_id: str):
    if patient_id not in STATE["patient_to_idx"]:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in cohort")

    idx = STATE["patient_to_idx"][patient_id]
    model = STATE["model"]
    graphs = STATE["graphs"]

    result = model.predict_subtype(*graphs)[idx]
    true_subtype = PAM50_SUBTYPES[STATE["y"][idx]]

    return PredictionResponse(
        patient_id=patient_id,
        true_subtype=true_subtype,
        predicted_subtype=result["predicted_subtype"],
        confidence=result["confidence"],
        all_probabilities=result["all_probabilities"],
        modality_contribution=result["modality_contribution"],
    )


@app.get("/api/explain/{patient_id}", response_model=ExplainResponse)
def explain(patient_id: str, layer: str = "rna", top_k: int = 10, steps: int = 20):
    """Gene-level attribution for one patient. `steps` defaults lower (20 vs the
    50 used in evaluation/attribution.py) to keep API latency reasonable for a
    live demo - reduces precision slightly but is much faster per request."""
    if patient_id not in STATE["patient_to_idx"]:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in cohort")
    if layer not in ("rna", "mirna", "meth"):
        raise HTTPException(status_code=400, detail="layer must be one of: rna, mirna, meth")

    idx = STATE["patient_to_idx"][patient_id]
    attribution, pred_class, _ = integrated_gradients(
        STATE["model"], STATE["graphs"], target_layer=layer, patient_idx=idx, steps=steps
    )

    feature_names = STATE["feature_names"][layer]

    top = top_attributed_features(attribution, feature_names, top_k=top_k)
    return ExplainResponse(
        patient_id=patient_id,
        layer=layer,
        top_genes=[GeneAttribution(gene=name, score=score) for name, score, _ in top],
    )


@app.get("/api/metrics")
def metrics():
    """Overall model performance for the dashboard. Returns None if
    metrics.json hasn't been generated yet - see metrics_seed.py."""
    if STATE["metrics"] is None:
        raise HTTPException(
            status_code=404, detail="Metrics not yet generated - run metrics_seed.py"
        )
    return STATE["metrics"]
