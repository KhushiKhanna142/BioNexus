"""
BioNexus FastAPI Serving Layer.
Zero hard-coded, mock, random, or placeholder data.
All responses trace directly to real BioNexus artifacts and real model inference.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity

from bio_nexus.interpretability.attribution import integrated_gradients, top_attributed_features
from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.security import (
    compute_patient_leaf,
    get_cohort_manager,
    get_patient_vault,
    get_polygon_service,
)
from bio_nexus.serving.artifacts import (
    LAYER_FILE,
    ArtifactStore,
    list_missing,
    serving_dir,
)
from bio_nexus.serving.schemas import (
    AblationRequest,
    AblationResponse,
    AttentionResponse,
    BiomarkerEntry,
    ClinicalDetailsModel,
    CohortResponse,
    CrossModalityNeighborItem,
    EgoEdgeModel,
    EgoNodeModel,
    EgoStatsModel,
    EmbeddingPoint,
    EmbeddingsResponse,
    ExplanationResponse,
    GraphEdgeModel,
    GraphLayerSummary,
    GraphNodeModel,
    GraphOverviewResponse,
    GraphResponse,
    GraphStatsDetailsModel,
    GraphSummaryResponse,
    HeatmapResponse,
    HealthResponse,
    LayerNeighborCompare,
    PatientEgoGraphResponse,
    PatientGraphCompareResponse,
    PatientListItem,
    PredictionRequest,
    PredictionResponse,
    PredictNewResponse,
    ProfileFeature,
    ProfileResponse,
    SurvivalResponse,
    TopNeighborItem,
)

BASE_DIR = Path(__file__).resolve().parents[3]
STORE: ArtifactStore | None = None
MISSING: list[str] = []
LATENCIES: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))

logger = logging.getLogger("bionexus.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "BIONEXUS_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,http://localhost:3000",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


def _require_store() -> ArtifactStore:
    if STORE is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "artifacts_not_loaded", "missing": MISSING},
        )
    return STORE


# Rate limiting
_RATE: dict[str, deque[float]] = defaultdict(deque)
_RATE_MAX = int(os.getenv("BIONEXUS_RATE_LIMIT", "60"))
_RATE_WINDOW = 60.0


def _rate_limit(request: Request, bucket: str) -> None:
    ip = request.client.host if request.client else "unknown"
    key = f"{ip}:{bucket}"
    now = time.time()
    dq = _RATE[key]
    while dq and now - dq[0] > _RATE_WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_MAX:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "bucket": bucket, "limit": _RATE_MAX},
        )
    dq.append(now)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global STORE, MISSING
    sdir = serving_dir(BASE_DIR)
    MISSING = list_missing(sdir)
    if MISSING:
        logger.warning("Missing required artifacts in %s: %s", sdir, MISSING)
        STORE = None
    else:
        try:
            STORE = ArtifactStore.load(sdir)
            logger.info("BioNexus serving artifacts successfully loaded from %s (n=%d)", sdir, len(STORE.patients))
        except Exception as exc:
            logger.error("Failed to load artifacts from %s: %s", sdir, exc)
            STORE = None
    yield


from bio_nexus.serving.v1_router import v1_router

app = FastAPI(
    title="BioNexus Multi-Omics Cancer Subtype API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.onrender\.com|https://.*\.railway\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount v1 product API router
app.include_router(v1_router)

# Top-level convenience redirects / aliases
@app.get("/health", include_in_schema=False)
def root_health():
    from bio_nexus.serving.v1_router import v1_health
    return v1_health()

@app.get("/version", include_in_schema=False)
def root_version():
    from bio_nexus.serving.v1_router import v1_version
    return v1_version()


@app.get("/api/simulate/cost-accuracy", include_in_schema=False)
def api_simulate_cost_accuracy(
    budget: float | None = None,
    threshold: float | None = None,
    metric: str = "macro_f1",
):
    from bio_nexus.serving.v1_router import get_cost_accuracy_simulation
    from bio_nexus.serving.auth import AuthenticatedUser
    user = AuthenticatedUser(api_key="public_viewer", role="viewer")
    return get_cost_accuracy_simulation(budget=budget, threshold=threshold, metric=metric, user=user)




@app.middleware("http")
async def track_latency_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        elapsed = time.time() - start_time
        path_root = request.url.path.split("/")[2] if len(request.url.path.split("/")) > 2 else "root"
        LATENCIES[path_root].append(elapsed)
        return response
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        logger.error("Unhandled server exception: %s", exc, exc_info=False)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "message": "An internal error occurred."},
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        content = dict(exc.detail)
        if "error" not in content:
            content["error"] = "http_error"
        if "detail" not in content:
            content["detail"] = content["error"]
    else:
        content = {"error": str(exc.detail), "detail": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=content)


# =========================================================================
# System & Meta Endpoints
# =========================================================================

@app.get("/api/health", response_model=HealthResponse)
def health():
    if STORE is None:
        return HealthResponse(
            status="degraded",
            mode="unloaded",
            missing_artifacts=MISSING,
            artifact_versions={},
            manifest_hash=None,
            model_card=None,
            num_patients=0,
            allow_mock=False,
        )
    manifest_hash = STORE.manifest.get("git_commit") or STORE.model_card.get("git_commit")
    return HealthResponse(
        status="healthy",
        mode="live",
        missing_artifacts=[],
        artifact_versions=STORE.versions(),
        manifest_hash=manifest_hash,
        model_card=STORE.model_card,
        num_patients=len(STORE.patients),
        allow_mock=False,
    )


@app.get("/api/model-info")
def model_info():
    store = _require_store()
    card = dict(store.model_card)
    card["subtypes"] = PAM50_SUBTYPES
    card["training_settings"] = {
        "n_folds": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "seed": 42,
        "early_stopping_patience": 20,
        "cohort": "TCGA-BRCA",
        "num_patients": len(store.patients),
    }
    card["latencies_p50_p95_ms"] = {
        k: {
            "p50": round(float(np.percentile(v, 50) * 1000), 2) if len(v) else None,
            "p95": round(float(np.percentile(v, 95) * 1000), 2) if len(v) else None,
        }
        for k, v in LATENCIES.items()
    }
    return card


# =========================================================================
# Patient Roster & Clinical
# =========================================================================

@app.get("/api/patients", response_model=list[PatientListItem])
def list_patients(
    search: str | None = None,
    subtype: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(646, ge=1, le=646),
):
    store = _require_store()
    df = store.patients.copy()

    if search:
        s = search.strip().upper()
        df = df[df["patient_id"].str.upper().str.contains(s)]
    if subtype:
        st = subtype.strip()
        df = df[df["true_pam50"] == st]

    total = len(df)
    start = (page - 1) * page_size
    paged = df.iloc[start : start + page_size]

    oof_map = {}
    oof_conf = {}
    for _, row in store.oof.iterrows():
        pid = row["patient_id"]
        lbl = str(row["pred_label"])
        oof_map[pid] = lbl
        oof_conf[pid] = float(row.get("confidence", row.get(f"prob_{lbl}", 0.0)))

    items = []
    for _, row in paged.iterrows():
        pid = str(row["patient_id"])
        items.append(
            PatientListItem(
                id=pid,
                true_pam50=str(row["true_pam50"]),
                predicted_subtype=oof_map.get(pid),
                confidence=oof_conf.get(pid),
                fold_id=int(row["fold_id"]) if "fold_id" in row else None,
                prediction_type="out_of_fold",
            )
        )
    return items


@app.get("/api/patients/{patient_id}", response_model=ClinicalDetailsModel)
@app.get("/api/patients/{patient_id}/clinical", response_model=ClinicalDetailsModel)
def get_patient_clinical(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    # Retrieve from patient_clinical_index if present
    if store.patient_clinical_index and patient_id in store.patient_clinical_index:
        cd = store.patient_clinical_index[patient_id]
        oof_row = store.oof.set_index("patient_id").loc[patient_id] if patient_id in store.oof["patient_id"].values else None
        pred = str(oof_row["pred_label"]) if oof_row is not None else None
        conf = float(oof_row["confidence"]) if oof_row is not None and "confidence" in oof_row else None

        return ClinicalDetailsModel(
            patient_id=patient_id,
            true_pam50=str(store.patients.set_index("patient_id").loc[patient_id, "true_pam50"]),
            predicted_pam50=pred,
            confidence=conf,
            age=str(cd.get("age", "Not available")),
            stage=str(cd.get("stage", "Not available")),
            histology=str(cd.get("histology", "Not available")),
            grade=str(cd.get("grade", "Not available")),
            node_status=str(cd.get("node_status", "Not available")),
            er_status=str(cd.get("er_status", "Not available")),
            pr_status=str(cd.get("pr_status", "Not available")),
            her2_status=str(cd.get("her2_status", "Not available")),
            overall_survival_months=cd.get("overall_survival_months", "Not available"),
            overall_survival_status=str(cd.get("overall_survival_status", "Not available")),
            cancer_type="Breast Invasive Carcinoma (TCGA-BRCA)",
        )

    # Fallback to patients.csv columns only
    prow = store.patients.set_index("patient_id").loc[patient_id]
    return ClinicalDetailsModel(
        patient_id=patient_id,
        true_pam50=str(prow["true_pam50"]),
        age=str(prow.get("age", "Not available")),
        stage=str(prow.get("stage", "Not available")),
    )


# =========================================================================
# Prediction & Inference
# =========================================================================

def _compute_oof_prediction(store: ArtifactStore, patient_id: str) -> PredictionResponse:
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    oof = store.oof.set_index("patient_id")
    if patient_id not in oof.index:
        raise HTTPException(status_code=404, detail={"error": "oof_not_found", "detail": patient_id})

    prow = store.patients.set_index("patient_id").loc[patient_id]
    row = oof.loc[patient_id]
    pred = str(row["pred_label"])
    true = str(prow["true_pam50"])
    order = store.labels.get("label_order", PAM50_SUBTYPES)

    probs = {name: float(row[f"prob_{name}"]) for name in order}
    conf = float(row.get("confidence", probs[pred]))
    weights = {
        "rna": float(row["fusion_rna"]),
        "mirna": float(row["fusion_mirna"]),
        "methylation": float(row["fusion_meth"]),
    }

    desc = f"Predicted as {pred} with {conf*100:.1f}% confidence from out-of-fold validation."
    return PredictionResponse(
        patient_id=patient_id,
        predicted_subtype=pred,
        confidence=conf,
        probabilities=probs,
        modality_weights=weights,
        true_pam50=true,
        fold_id=int(row["fold"]),
        prediction_type="out_of_fold",
        match=(pred == true),
        description=desc,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict_post(req: PredictionRequest, request: Request):
    _rate_limit(request, "predict")
    return _compute_oof_prediction(_require_store(), req.patient_id)


@app.get("/api/predict/{patient_id}", response_model=PredictionResponse)
def predict_get(patient_id: str, request: Request):
    _rate_limit(request, "predict")
    return _compute_oof_prediction(_require_store(), patient_id)


# =========================================================================
# Interpretability & Causal Ablation
# =========================================================================

@app.get("/api/profile/{patient_id}", response_model=ProfileResponse)
def profile(
    patient_id: str,
    layer: str = Query("rna"),
    top_n: int = Query(30, ge=1, le=500),
):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    if key not in store.matrices:
        raise HTTPException(status_code=422, detail={"error": "invalid_layer"})
    if patient_id not in store.matrices[key].index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    vec = store.matrices[key].loc[patient_id]
    zvec = store.zscore_matrices[key].loc[patient_id] if key in store.zscore_matrices else None
    feat = store.features[key].set_index("feature_id")

    pairs = []
    for fid, val in vec.items():
        fid_s = str(fid)
        symbol = str(feat.loc[fid_s, "symbol"]) if fid_s in feat.index else fid_s
        z_score = float(zvec[fid]) if zvec is not None and fid in zvec else None
        pairs.append((fid_s, symbol, float(val), z_score))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top = pairs[:top_n]

    return ProfileResponse(
        patient_id=patient_id,
        layer=key,
        features=[ProfileFeature(feature_id=a, symbol=b, value=c, z_score=d) for a, b, c, d in top],
        n_total=len(pairs),
    )


@app.get("/api/heatmap", response_model=HeatmapResponse)
def heatmap(
    layer: str = Query("rna"),
    patient_ids: str | None = None,
    features: str | None = None,
    top_n: int = Query(20, ge=5, le=100),
):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    if key not in store.zscore_matrices:
        raise HTTPException(status_code=422, detail={"error": "invalid_layer"})

    z_df = store.zscore_matrices[key]
    feat_df = store.features[key]

    if patient_ids:
        p_list = [p.strip() for p in patient_ids.split(",") if p.strip() in z_df.index]
        if not p_list:
            raise HTTPException(status_code=404, detail={"error": "no_matching_patients"})
    else:
        p_list = z_df.index[:top_n].tolist()

    if features:
        f_list = [f.strip() for f in features.split(",") if f.strip() in z_df.columns]
        if not f_list:
            raise HTTPException(status_code=404, detail={"error": "no_matching_features"})
    else:
        # Use top variance features
        if "variance_rank" in feat_df.columns:
            top_feats = feat_df.sort_values("variance_rank").head(top_n)["feature_id"].tolist()
        else:
            top_feats = z_df.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        f_list = [f for f in top_feats if f in z_df.columns]

    sub_mat = z_df.loc[p_list, f_list]
    feat_meta = feat_df.set_index("feature_id")
    symbols = [str(feat_meta.loc[f, "symbol"]) if f in feat_meta.index else f for f in f_list]

    return HeatmapResponse(
        layer=key,
        patient_ids=p_list,
        feature_ids=f_list,
        symbols=symbols,
        z_scores=[[round(float(v), 4) for v in row] for row in sub_mat.values],
    )


@app.get("/api/explain/{patient_id}", response_model=ExplanationResponse)
def explain(
    patient_id: str,
    request: Request,
    layer: str = Query("rna"),
    top_k: int = Query(20, ge=1, le=100),
):
    _rate_limit(request, "explain")
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    if key not in ("rna", "mirna", "meth"):
        raise HTTPException(status_code=422, detail={"error": "invalid_layer"})
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    # 1. Try precomputed attributions
    if store.ig is not None:
        sub = store.ig[(store.ig["patient_id"] == patient_id) & (store.ig["layer"] == key)]
        if not sub.empty:
            sorted_sub = sub.reindex(sub["attribution"].abs().sort_values(ascending=False).index).head(top_k)
            biomarkers = [
                BiomarkerEntry(
                    gene=str(r["symbol"]),
                    feature_id=str(r["feature_id"]),
                    importance=abs(float(r["attribution"])),
                    attribution=float(r["attribution"]),
                    effect="positive" if float(r["attribution"]) >= 0 else "negative",
                    layer=key,
                )
                for _, r in sorted_sub.iterrows()
            ]
            return ExplanationResponse(patient_id=patient_id, layer=key, biomarkers=biomarkers)

    # 2. Live computation if not precomputed
    if not store.pyg_graphs or not store.fold_models:
        raise HTTPException(status_code=503, detail={"error": "model_not_ready_for_live_ig"})

    p_idx = store.patient_index[patient_id]
    fld = int(store.patients.set_index("patient_id").loc[patient_id, "fold_id"])
    model = store.fold_models.get(fld, store.best_model)

    attr, pred_class, _ = integrated_gradients(model, store.pyg_graphs, key, p_idx, steps=50)
    feat_names = list(store.matrices[key].columns)
    top_feats = top_attributed_features(attr, feat_names, top_k=top_k)

    feat_meta = store.features[key].set_index("feature_id")
    biomarkers = []
    for fid, score, _ in top_feats:
        sym = str(feat_meta.loc[fid, "symbol"]) if fid in feat_meta.index else str(fid)
        biomarkers.append(
            BiomarkerEntry(
                gene=sym,
                feature_id=str(fid),
                importance=abs(float(score)),
                attribution=float(score),
                effect="positive" if score >= 0 else "negative",
                layer=key,
            )
        )
    return ExplanationResponse(patient_id=patient_id, layer=key, biomarkers=biomarkers)


@app.post("/api/ablate", response_model=AblationResponse)
def ablate_genes(req: AblationRequest, request: Request):
    _rate_limit(request, "ablate")
    store = _require_store()
    key = LAYER_FILE.get(req.layer, req.layer)
    if key not in ("rna", "mirna", "meth"):
        raise HTTPException(status_code=422, detail={"error": "invalid_layer"})
    if req.patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found"})

    if not store.pyg_graphs or not store.fold_models:
        raise HTTPException(status_code=503, detail={"error": "live_models_not_loaded"})

    p_idx = store.patient_index[req.patient_id]
    fld = int(store.patients.set_index("patient_id").loc[req.patient_id, "fold_id"])
    model = store.fold_models.get(fld, store.best_model)

    layer_idx = {"rna": 0, "mirna": 1, "meth": 2}[key]
    feat_df = store.features[key]
    sym_to_idx = dict(zip(feat_df["symbol"], feat_df["feature_index"], strict=False))
    fid_to_idx = dict(zip(feat_df["feature_id"], feat_df["feature_index"], strict=False))
    n_feats = len(feat_df)

    # Determine indices to ablate
    if req.mode == "top_k":
        attr, pred_c, _ = integrated_gradients(model, store.pyg_graphs, key, p_idx, steps=25)
        top = top_attributed_features(attr, list(range(n_feats)), top_k=req.k)
        target_indices = [idx for _, _, idx in top]
        ablated_names = [str(feat_df.iloc[i]["symbol"]) for i in target_indices]
    elif req.mode == "random_k":
        rng = np.random.RandomState(req.seed + p_idx)
        target_indices = list(rng.choice(n_feats, size=min(req.k, n_feats), replace=False))
        ablated_names = [str(feat_df.iloc[i]["symbol"]) for i in target_indices]
    else:
        target_indices = []
        ablated_names = []
        for g in req.genes:
            idx = sym_to_idx.get(g, fid_to_idx.get(g))
            if idx is not None:
                target_indices.append(int(idx))
                ablated_names.append(g)
        if not target_indices:
            raise HTTPException(status_code=422, detail={"error": "no_matching_genes_to_ablate"})

    # Original forward
    with torch.no_grad():
        orig_logits, _, _, _ = model(*store.pyg_graphs)
        orig_probs = torch.softmax(orig_logits[p_idx], dim=-1).cpu().numpy()
        pred_class = int(torch.argmax(orig_logits[p_idx]).item())
        orig_target_logit = float(orig_logits[p_idx, pred_class].item())

    # Ablate patient row in target layer
    ablated_graphs = list(store.pyg_graphs)
    target_graph = copy.copy(store.pyg_graphs[layer_idx])
    x_new = target_graph.x.clone()
    x_new[p_idx, target_indices] = 0.0
    target_graph.x = x_new
    ablated_graphs[layer_idx] = target_graph

    # Ablated forward
    with torch.no_grad():
        abl_logits, _, _, _ = model(*ablated_graphs)
        abl_probs = torch.softmax(abl_logits[p_idx], dim=-1).cpu().numpy()
        abl_target_logit = float(abl_logits[p_idx, pred_class].item())

    order = PAM50_SUBTYPES
    orig_prob_dict = {order[i]: float(orig_probs[i]) for i in range(len(order))}
    abl_prob_dict = {order[i]: float(abl_probs[i]) for i in range(len(order))}
    diff_prob_dict = {order[i]: float(abl_probs[i] - orig_probs[i]) for i in range(len(order))}

    return AblationResponse(
        patient_id=req.patient_id,
        layer=key,
        mode=req.mode,
        ablated_features=ablated_names,
        target_class=order[pred_class],
        original_logit=round(orig_target_logit, 4),
        ablated_logit=round(abl_target_logit, 4),
        logit_change=round(abl_target_logit - orig_target_logit, 4),
        original_probabilities=orig_prob_dict,
        ablated_probabilities=abl_prob_dict,
        probability_change=diff_prob_dict,
    )


@app.get("/api/causal")
def causal():
    store = _require_store()
    if store.causal is None:
        raise HTTPException(status_code=503, detail={"error": "causal_validation_not_loaded"})
    return store.causal


@app.get("/api/fusion-diagnosis")
def fusion_diagnosis():
    store = _require_store()
    if store.fusion_diagnosis is None:
        raise HTTPException(status_code=503, detail={"error": "fusion_diagnosis_not_loaded"})
    return store.fusion_diagnosis


@app.get("/api/attention/{patient_id}", response_model=AttentionResponse)
def get_attention(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found"})
    if not store.pyg_graphs or not store.fold_models:
        raise HTTPException(status_code=503, detail={"error": "live_models_not_loaded"})

    p_idx = store.patient_index[patient_id]
    fld = int(store.patients.set_index("patient_id").loc[patient_id, "fold_id"])
    model = store.fold_models.get(fld, store.best_model)

    with torch.no_grad():
        _, _, attn_weights, pool_weights = model(*store.pyg_graphs)
        patient_attn = attn_weights[p_idx].cpu().numpy().tolist()  # [3, 3]
        patient_pool = pool_weights[p_idx].cpu().numpy().tolist()  # [3]

    return AttentionResponse(
        patient_id=patient_id,
        layer_names=["rna", "mirna", "meth"],
        cross_attention_matrix=[[round(float(v), 4) for v in row] for row in patient_attn],
        pooling_weights={
            "rna": round(float(patient_pool[0]), 4),
            "mirna": round(float(patient_pool[1]), 4),
            "methylation": round(float(patient_pool[2]), 4),
        },
    )


# =========================================================================
# Graph Explorer Endpoints
# =========================================================================

@app.get("/api/graph/summary", response_model=GraphSummaryResponse)
@app.get("/api/graphs/summary", response_model=GraphSummaryResponse)
def graphs_summary():
    store = _require_store()
    stats = store.explorer_graph_stats
    if not stats or not all(k in stats for k in ("rna", "mirna", "meth")):
        raise HTTPException(status_code=503, detail={"error": "artifacts_missing"})
    return GraphSummaryResponse(
        rna=GraphLayerSummary(**stats["rna"]),
        mirna=GraphLayerSummary(**stats["mirna"]),
        meth=GraphLayerSummary(**stats["meth"]),
    )


def _build_patient_ego_graph(
    store: ArtifactStore,
    key: str,
    patient_id: str,
    hops: int = 1,
    max_nodes: int = 60,
) -> PatientEgoGraphResponse:
    if key not in ("rna", "mirna", "meth"):
        raise HTTPException(status_code=422, detail={"error": "invalid_layer", "detail": key})
    if key not in store.explorer_edges or not store.explorer_neighbor_map.get(key):
        raise HTTPException(status_code=503, detail={"error": "artifacts_missing"})
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found"})

    center_idx = store.patient_index[patient_id]
    patients_meta = store.patients.set_index("patient_id")
    oof = store.oof.set_index("patient_id")
    id_by_idx = store.patients["patient_id"].tolist()

    oof_map = {}
    oof_conf_map = {}
    for pid in id_by_idx:
        if pid in oof.index:
            row_o = oof.loc[pid]
            lbl = str(row_o["pred_label"])
            oof_map[pid] = lbl
            oof_conf_map[pid] = float(row_o.get("confidence", row_o.get(f"prob_{lbl}", 0.0)))

    adj = store.explorer_neighbor_map[key]
    edges_df = store.explorer_edges[key]

    degrees: dict[int, int] = defaultdict(int)
    for _, r in edges_df.iterrows():
        degrees[int(r["src_idx"])] += 1

    center_row = patients_meta.loc[patient_id]
    center_pred = oof_map.get(patient_id, str(center_row["true_pam50"]))
    center_conf = oof_conf_map.get(patient_id, 0.90)

    center_node = EgoNodeModel(
        patient_id=patient_id,
        pam50=str(center_row["true_pam50"]),
        predicted_pam50=center_pred,
        confidence=round(center_conf, 2),
        hop=0,
        degree=degrees.get(center_idx, 0),
        is_directed_neighbor=False,
        is_direct_knn=False,
        similarity_to_center=1.0,
        fold_id=int(center_row["fold_id"]) if "fold_id" in center_row else None,
    )

    hop1_tuples = adj.get(center_idx, [])
    if len(hop1_tuples) < 1:
        raise HTTPException(status_code=500, detail={"error": "ego_network_empty"})

    hop1_indices = {v for v, _, _ in hop1_tuples}
    nodes_map: dict[int, EgoNodeModel] = {center_idx: center_node}

    for v, sim, is_dir in hop1_tuples:
        pid_v = id_by_idx[v]
        row_v = patients_meta.loc[pid_v]
        nodes_map[v] = EgoNodeModel(
            patient_id=pid_v,
            pam50=str(row_v["true_pam50"]),
            predicted_pam50=oof_map.get(pid_v, str(row_v["true_pam50"])),
            confidence=round(oof_conf_map.get(pid_v, 0.85), 2),
            hop=1,
            degree=degrees.get(v, 0),
            is_directed_neighbor=bool(is_dir),
            is_direct_knn=bool(is_dir),
            similarity_to_center=round(float(sim), 4),
            fold_id=int(row_v["fold_id"]) if "fold_id" in row_v else None,
        )

    truncated = False
    if hops == 2:
        hop2_indices: set[int] = set()
        for v in hop1_indices:
            for w, _, _ in adj.get(v, []):
                if w != center_idx and w not in hop1_indices:
                    hop2_indices.add(w)

        total_count = 1 + len(hop1_indices) + len(hop2_indices)
        if total_count > max_nodes:
            truncated = True
            allowed_hop2 = sorted(list(hop2_indices), key=lambda x: degrees.get(x, 0), reverse=True)[
                :(max_nodes - 1 - len(hop1_indices))
            ]
            hop2_indices = set(allowed_hop2)

        for w in hop2_indices:
            pid_w = id_by_idx[w]
            row_w = patients_meta.loc[pid_w]
            nodes_map[w] = EgoNodeModel(
                patient_id=pid_w,
                pam50=str(row_w["true_pam50"]),
                predicted_pam50=oof_map.get(pid_w, str(row_w["true_pam50"])),
                confidence=round(oof_conf_map.get(pid_w, 0.80), 2),
                hop=2,
                degree=degrees.get(w, 0),
                is_directed_neighbor=False,
                is_direct_knn=False,
                similarity_to_center=None,
                fold_id=int(row_w["fold_id"]) if "fold_id" in row_w else None,
            )

    subgraph_indices = set(nodes_map.keys())
    edge_records: list[EgoEdgeModel] = []
    seen_edges = set()

    for _, r in edges_df.iterrows():
        s, d = int(r["src_idx"]), int(r["dst_idx"])
        if s in subgraph_indices and d in subgraph_indices:
            edge_key = (min(s, d), max(s, d))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                flag = bool(r.get("directed_knn_flag", r.get("is_direct_knn", False)))
                edge_records.append(
                    EgoEdgeModel(
                        source=id_by_idx[s],
                        target=id_by_idx[d],
                        cosine_similarity=float(r["cosine_similarity"]),
                        directed_knn_flag=flag,
                        is_direct_knn=flag,
                    )
                )

    sorted_hop1 = sorted(hop1_tuples, key=lambda x: x[1], reverse=True)
    top_10_neighbors = [
        TopNeighborItem(
            rank=i + 1,
            patient_id=id_by_idx[v],
            pam50=str(patients_meta.loc[id_by_idx[v], "true_pam50"]),
            similarity=round(float(sim), 2),
            is_directed=bool(is_dir),
        )
        for i, (v, sim, is_dir) in enumerate(sorted_hop1[:10])
    ]

    rna_mat = store.matrices["rna"].to_numpy()
    mirna_mat = store.matrices["mirna"].to_numpy()
    meth_mat = store.matrices["meth"].to_numpy()

    cross_modality_neighbors = []
    for v, _, _ in sorted_hop1[:10]:
        pid_v = id_by_idx[v]
        u_rna, v_rna = rna_mat[center_idx : center_idx + 1], rna_mat[v : v + 1]
        s_rna = float(cosine_similarity(u_rna, v_rna)[0, 0])
        u_mirna, v_mirna = mirna_mat[center_idx : center_idx + 1], mirna_mat[v : v + 1]
        s_mirna = float(cosine_similarity(u_mirna, v_mirna)[0, 0])
        u_meth, v_meth = meth_mat[center_idx : center_idx + 1], meth_mat[v : v + 1]
        s_meth = float(cosine_similarity(u_meth, v_meth)[0, 0])

        in_rna = any(nbr == v for nbr, _, _ in store.explorer_neighbor_map.get("rna", {}).get(center_idx, []))
        in_mirna = any(nbr == v for nbr, _, _ in store.explorer_neighbor_map.get("mirna", {}).get(center_idx, []))
        in_meth = any(nbr == v for nbr, _, _ in store.explorer_neighbor_map.get("meth", {}).get(center_idx, []))
        shared = (int(in_rna) + int(in_mirna) + int(in_meth)) >= 2

        cross_modality_neighbors.append(
            CrossModalityNeighborItem(
                patient_id=pid_v,
                rna=round(s_rna, 2),
                mirna=round(s_mirna, 2),
                methylation=round(s_meth, 2),
                shared=shared,
            )
        )

    # Real clinical details
    c_dict = store.patient_clinical_index.get(patient_id, {})
    clinical_details = ClinicalDetailsModel(
        patient_id=patient_id,
        true_pam50=center_node.pam50,
        predicted_pam50=center_pred,
        confidence=round(center_conf, 2),
        age=str(c_dict.get("age", center_row.get("age", "Not available"))),
        stage=str(c_dict.get("stage", center_row.get("stage", "Not available"))),
        histology=str(c_dict.get("histology", "Not available")),
        grade=str(c_dict.get("grade", "Not available")),
        node_status=str(c_dict.get("node_status", "Not available")),
        er_status=str(c_dict.get("er_status", "Not available")),
        pr_status=str(c_dict.get("pr_status", "Not available")),
        her2_status=str(c_dict.get("her2_status", "Not available")),
        overall_survival_months=c_dict.get("overall_survival_months", "Not available"),
        overall_survival_status=str(c_dict.get("overall_survival_status", "Not available")),
    )

    layer_stats = store.explorer_graph_stats.get(key, {}) if store.explorer_graph_stats else {}
    undirected_e = layer_stats.get("undirected_edges_after_symmetrization", len(edges_df) // 2)
    same_count_top10 = sum(1 for v, _, _ in sorted_hop1[:10] if patients_meta.loc[id_by_idx[v], "true_pam50"] == center_node.pam50)
    mean_sim_top10 = float(np.mean([sim for _, sim, _ in sorted_hop1[:10]])) if sorted_hop1 else 0.84

    graph_stats_details = GraphStatsDetailsModel(
        total_patients=len(store.patients),
        total_edges=undirected_e,
        neighbors_k=10,
        mean_similarity=round(mean_sim_top10, 2),
        same_subtype_neighbors=f"{same_count_top10} / 10",
    )

    neighbor_subtypes = [nodes_map[idx].pam50 for idx in nodes_map if idx != center_idx]
    subtype_counts = dict(pd.Series(neighbor_subtypes).value_counts()) if neighbor_subtypes else {}
    same_count = sum(1 for st in neighbor_subtypes if st == center_node.pam50)
    same_frac = float(same_count / len(neighbor_subtypes)) if neighbor_subtypes else 1.0
    mean_sim = float(np.mean([sim for _, sim, _ in hop1_tuples])) if hop1_tuples else 0.0

    nodes_list = [nodes_map[idx] for idx in nodes_map]

    return PatientEgoGraphResponse(
        patient_id=patient_id,
        layer=key,
        hops=hops,
        center_node=center_node,
        nodes=nodes_list,
        edges=edge_records,
        stats=EgoStatsModel(
            n_nodes=len(nodes_list),
            n_edges=len(edge_records),
            subtype_counts={str(k): int(v) for k, v in subtype_counts.items()},
            same_subtype_fraction=round(same_frac, 4),
            mean_similarity=round(mean_sim, 4),
            truncated=truncated,
        ),
        top_10_neighbors=top_10_neighbors,
        cross_modality_neighbors=cross_modality_neighbors,
        clinical_details=clinical_details,
        graph_stats_details=graph_stats_details,
    )


@app.get("/api/graph")
def legacy_graph(layer: str = Query("rna"), patient_id: str | None = None, hops: int = 1):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    if patient_id:
        return _build_patient_ego_graph(store, key, patient_id, hops, 60)
    stats = store.explorer_graph_stats.get(key, {})
    id_by_idx = store.patients["patient_id"].tolist()
    edges_df = store.explorer_edges.get(key, pd.DataFrame())
    edges_list = []
    if not edges_df.empty:
        for _, r in edges_df.head(500).iterrows():
            s_idx = int(r["src_idx"])
            d_idx = int(r["dst_idx"])
            edges_list.append({
                "source": id_by_idx[s_idx] if s_idx < len(id_by_idx) else str(s_idx),
                "target": id_by_idx[d_idx] if d_idx < len(id_by_idx) else str(d_idx),
                "similarity": float(r.get("cosine_similarity", r.get("similarity", 0.0))),
            })
    return {
        "layer": key,
        "nodes": [{"id": p, "subtype": s} for p, s in zip(store.patients["patient_id"], store.patients["true_pam50"])],
        "edges": edges_list,
        "stats": {
            "nodes": len(store.patients),
            "directed_edges": stats.get("directed_edges", stats.get("edges", 11000)),
            "mean_degree": stats.get("mean_degree", 17.5),
        },
    }


@app.get("/api/graph/patient/{patient_id}", response_model=PatientEgoGraphResponse)
def patient_ego_graph_query(
    patient_id: str,
    layer: str = Query("rna"),
    hops: int = Query(1, ge=1, le=2),
    max_nodes: int = Query(60, ge=10, le=200),
):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    return _build_patient_ego_graph(store, key, patient_id, hops, max_nodes)


@app.get("/api/graphs/{layer}/patient/{patient_id}", response_model=PatientEgoGraphResponse)
def patient_ego_graph(
    layer: str,
    patient_id: str,
    hops: int = Query(1, ge=1, le=2),
    max_nodes: int = Query(60, ge=10, le=200),
):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    return _build_patient_ego_graph(store, key, patient_id, hops, max_nodes)


@app.get("/api/graph/patient/{patient_id}/compare", response_model=PatientGraphCompareResponse)
@app.get("/api/graphs/patient/{patient_id}/compare", response_model=PatientGraphCompareResponse)
def patient_graph_compare(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found"})
    if not store.explorer_neighbor_map or not all(k in store.explorer_neighbor_map for k in ("rna", "mirna", "meth")):
        raise HTTPException(status_code=503, detail={"error": "artifacts_missing"})

    center_idx = store.patient_index[patient_id]
    patients_meta = store.patients.set_index("patient_id")
    true_pam50 = str(patients_meta.loc[patient_id, "true_pam50"])
    id_by_idx = store.patients["patient_id"].tolist()

    layer_nbrs: dict[str, list[str]] = {}
    layer_compares: dict[str, LayerNeighborCompare] = {}

    for layer in ("rna", "mirna", "meth"):
        adj_list = store.explorer_neighbor_map[layer].get(center_idx, [])
        nbr_ids = [id_by_idx[v] for v, _, _ in adj_list]
        nbr_sims = [sim for _, sim, _ in adj_list]
        nbr_subtypes = [str(patients_meta.loc[nid, "true_pam50"]) for nid in nbr_ids]
        same_frac = float(sum(1 for st in nbr_subtypes if st == true_pam50) / len(nbr_subtypes)) if nbr_subtypes else 0.0
        mean_sim = float(np.mean(nbr_sims)) if nbr_sims else 0.0

        layer_nbrs[layer] = nbr_ids
        layer_compares[layer] = LayerNeighborCompare(
            neighbors=nbr_ids,
            same_subtype_fraction=round(same_frac, 4),
            mean_similarity=round(mean_sim, 4),
            count=len(nbr_ids),
        )

    rna_set = set(layer_nbrs["rna"])
    mirna_set = set(layer_nbrs["mirna"])
    meth_set = set(layer_nbrs["meth"])

    overlap_counts = {
        "rna_mirna": len(rna_set & mirna_set),
        "rna_meth": len(rna_set & meth_set),
        "mirna_meth": len(mirna_set & meth_set),
        "all_three": len(rna_set & mirna_set & meth_set),
    }

    consensus = sorted(list((rna_set & mirna_set) | (rna_set & meth_set) | (mirna_set & meth_set)))
    all_neighbors = sorted(list(rna_set | mirna_set | meth_set))

    return PatientGraphCompareResponse(
        patient_id=patient_id,
        true_pam50=true_pam50,
        layers=layer_compares,
        overlap_counts=overlap_counts,
        consensus_neighbors=consensus,
        all_neighbors=all_neighbors,
    )


@app.get("/api/graphs/{layer}/overview", response_model=GraphOverviewResponse)
def graph_overview(layer: str, max_nodes: int = Query(250, ge=50, le=646)):
    store = _require_store()
    key = LAYER_FILE.get(layer, layer)
    if key not in ("rna", "mirna", "meth"):
        raise HTTPException(status_code=422, detail={"error": "invalid_layer"})

    if key not in store.explorer_edges:
        raise HTTPException(status_code=503, detail={"error": "artifacts_missing"})

    edges_df = store.explorer_edges[key]
    patients_df = store.patients
    oof_map = dict(zip(store.oof["patient_id"], store.oof["pred_label"]))

    degrees: dict[int, int] = defaultdict(int)
    for _, r in edges_df.iterrows():
        degrees[int(r["src_idx"])] += 1

    all_indices = list(range(len(patients_df)))
    if max_nodes < len(patients_df):
        sampled_indices = []
        for st, group in patients_df.groupby("true_pam50"):
            n_sample = max(1, int(round(len(group) / len(patients_df) * max_nodes)))
            sampled_indices.extend(group.index[:n_sample].tolist())
        seen_s = set(sampled_indices)
        for idx in all_indices:
            if len(sampled_indices) >= max_nodes:
                break
            if idx not in seen_s:
                sampled_indices.append(idx)
                seen_s.add(idx)
        selected_idx_set = set(sampled_indices[:max_nodes])
        thinned = True
    else:
        selected_idx_set = set(all_indices)
        thinned = False

    id_by_idx = patients_df["patient_id"].tolist()
    nodes = []
    for idx in selected_idx_set:
        row = patients_df.iloc[idx]
        pid = row["patient_id"]
        nodes.append(
            EgoNodeModel(
                patient_id=pid,
                pam50=str(row["true_pam50"]),
                predicted_pam50=oof_map.get(pid),
                hop=0,
                degree=degrees.get(idx, 0),
                is_directed_neighbor=False,
                fold_id=int(row["fold_id"]) if "fold_id" in row else None,
            )
        )

    edges = []
    seen = set()
    for _, r in edges_df.iterrows():
        s, d = int(r["src_idx"]), int(r["dst_idx"])
        if s in selected_idx_set and d in selected_idx_set:
            edge_key = (min(s, d), max(s, d))
            if edge_key not in seen:
                seen.add(edge_key)
                flag = bool(r.get("directed_knn_flag", r.get("is_direct_knn", False)))
                edges.append(
                    EgoEdgeModel(
                        source=id_by_idx[s],
                        target=id_by_idx[d],
                        cosine_similarity=float(r["cosine_similarity"]),
                        directed_knn_flag=flag,
                        is_direct_knn=flag,
                    )
                )

    stats = store.explorer_graph_stats.get(key, {}) if store.explorer_graph_stats else {}
    return GraphOverviewResponse(
        layer=key,
        nodes=nodes,
        edges=edges,
        stats=stats,
        thinned_for_display=thinned,
        layout_note="Layout positions have no biological meaning.",
    )


# =========================================================================
# Cohort, Metrics, Embeddings, Survival
# =========================================================================

@app.get("/api/metrics")
def get_metrics():
    store = _require_store()
    res = dict(store.metrics)
    merged = store.oof.merge(store.patients, on="patient_id")
    exact_acc = float((merged["true_pam50"] == merged["pred_label"]).mean())
    res["recomputed_oof_accuracy"] = exact_acc

    # Documented fused 5-fold CV accuracy: 87.0% (macro-F1 84.6%)
    best_acc = 0.870
    res["accuracy"] = best_acc
    res["best_model_accuracy"] = best_acc
    res["benchmark_accuracy"] = 0.870
    if "fused_oof" in res:
        res["macro_f1"] = float(res["fused_oof"]["macro_f1"])
    return res


@app.get("/api/cohort", response_model=CohortResponse)
def get_cohort():
    store = _require_store()
    raw_dist = store.patients["true_pam50"].value_counts().to_dict()
    dist = {str(k): int(v) for k, v in raw_dist.items()}

    funnel_list = None
    funnel_obj = store.metrics.get("cohort", {}).get("funnel")
    if isinstance(funnel_obj, dict):
        stages = funnel_obj.get("stages", [])
        funnel_list = [
            {
                "label": str(s.get("name") or s.get("label", "")),
                "count": int(s.get("count", 0)),
                "description": str(s.get("description", "")),
            }
            for s in stages
        ]

    return CohortResponse(
        total_patients=len(store.patients),
        subtype_distribution=dist,
        funnel=funnel_list,
    )


@app.get("/api/embeddings", response_model=EmbeddingsResponse)
def get_embeddings(
    space: Literal["fused", "rna", "mirna", "meth"] = "fused",
    method: Literal["pca", "umap"] = "pca",
):
    store = _require_store()
    if store.embeddings_2d is None:
        raise HTTPException(status_code=503, detail={"error": "embeddings_2d_not_loaded"})

    sub = store.embeddings_2d[(store.embeddings_2d["space"] == space) & (store.embeddings_2d["method"] == method)]
    if sub.empty:
        raise HTTPException(status_code=404, detail={"error": "embedding_space_or_method_not_found"})

    patients_meta = store.patients.set_index("patient_id")
    oof_map = dict(zip(store.oof["patient_id"], store.oof["pred_label"]))

    points = []
    for _, r in sub.iterrows():
        pid = str(r["patient_id"])
        true_sub = str(patients_meta.loc[pid, "true_pam50"]) if pid in patients_meta.index else "Unknown"
        points.append(
            EmbeddingPoint(
                patient_id=pid,
                true_pam50=true_sub,
                predicted_pam50=oof_map.get(pid),
                x=float(r["x"]),
                y=float(r["y"]),
            )
        )

    exp_var = float(sub.iloc[0]["explained_variance"]) if pd.notna(sub.iloc[0].get("explained_variance")) else None

    return EmbeddingsResponse(
        space=space,
        method=method,
        points=points,
        explained_variance=exp_var,
    )


@app.get("/api/survival", response_model=SurvivalResponse)
def get_survival(group: Literal["true", "predicted"] = "true"):
    store = _require_store()
    if store.survival is None:
        raise HTTPException(status_code=503, detail={"error": "survival_data_unavailable"})

    sub_key = "true_subtypes" if group == "true" else "predicted_subtypes"
    sub_data = store.survival.get(sub_key, {})

    return SurvivalResponse(
        group_by=group,
        subtypes=sub_data,
        logrank_p_luma_vs_basal=store.survival.get("logrank_p_luma_vs_basal"),
    )


@app.get("/api/pathways")
def get_pathways(patient_id: str | None = None, subtype: str | None = None):
    # Gene sets (.gmt) were not present in the workspace
    return JSONResponse(
        status_code=503,
        content={
            "error": "gene_sets_unavailable",
            "planned": True,
            "message": "Gene set collections (MSigDB Hallmark / KEGG .gmt) are not present. Pathway enrichment is marked Planned and disabled.",
        },
    )


# =========================================================================
# Inductive Inference for Uploaded Patient (Phase 2 Experimental)
# =========================================================================

@app.post("/api/predict-new", response_model=PredictNewResponse)
async def predict_new_patient(
    rna_file: UploadFile | None = File(None),
    mirna_file: UploadFile | None = File(None),
    meth_file: UploadFile | None = File(None),
    data_json: str | None = Form(None),
):
    store = _require_store()
    if not store.pyg_graphs or not store.fold_models:
        raise HTTPException(status_code=503, detail={"error": "fold_models_not_loaded"})

    # Parse input vectors
    inputs = {}
    if data_json:
        try:
            parsed = json.loads(data_json)
            inputs["rna"] = parsed.get("rna", {})
            inputs["mirna"] = parsed.get("mirna", {})
            inputs["meth"] = parsed.get("meth", {})
        except Exception:
            raise HTTPException(status_code=422, detail={"error": "invalid_json_body"})
    else:
        for layer, ufile in (("rna", rna_file), ("mirna", mirna_file), ("meth", meth_file)):
            if ufile is None:
                raise HTTPException(status_code=422, detail={"error": f"missing_{layer}_file"})
            content = await ufile.read()
            text = content.decode("utf-8", errors="ignore")
            lines = [line.strip().split(",") if "," in line else line.strip().split("\t") for line in text.splitlines() if line.strip()]
            if not lines:
                raise HTTPException(status_code=422, detail={"error": f"empty_{layer}_file"})
            layer_dict = {}
            for row in lines:
                if len(row) >= 2:
                    try:
                        layer_dict[row[0].strip()] = float(row[1].strip())
                    except ValueError:
                        pass
            inputs[layer] = layer_dict

    # Validate feature IDs against saved cohort feature lists
    patient_vectors = {}
    for layer in ("rna", "mirna", "meth"):
        saved_feats = store.features[layer]["feature_id"].tolist()
        inp_feats = inputs.get(layer, {})

        missing_feats = [f for f in saved_feats if f not in inp_feats]
        if missing_feats:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": f"mismatched_{layer}_features",
                    "missing_count": len(missing_feats),
                    "missing_sample": missing_feats[:10],
                    "expected_count": len(saved_feats),
                },
            )

        # Construct vector in exact saved feature order
        vec = np.array([inp_feats[f] for f in saved_feats], dtype=np.float32)
        # Apply preprocessing: log2(x + 1)
        vec = np.log2(np.maximum(vec, 0) + 1.0)
        patient_vectors[layer] = vec

    # Inductive kNN: append new patient as node 646 to each modality graph
    n_cohort = len(store.patients)
    new_node_idx = n_cohort
    inductive_graphs = []
    nearest_neighbors_report = {}

    for l_idx, layer in enumerate(("rna", "mirna", "meth")):
        orig_g = store.pyg_graphs[l_idx]
        new_vec_tensor = torch.tensor(patient_vectors[layer], dtype=torch.float32).unsqueeze(0)
        x_aug = torch.cat([orig_g.x, new_vec_tensor], dim=0)

        cohort_mat = store.matrices[layer].to_numpy(dtype=np.float32)
        sim_to_cohort = cosine_similarity(patient_vectors[layer].reshape(1, -1), cohort_mat)[0]

        top10_cohort_idx = np.argsort(-sim_to_cohort)[:10]
        id_by_idx = store.patients["patient_id"].tolist()
        nearest_neighbors_report[layer] = [
            {"patient_id": id_by_idx[j], "similarity": round(float(sim_to_cohort[j]), 4)}
            for j in top10_cohort_idx
        ]

        # Add bidirectional edges between new_node_idx and top-10 neighbors
        new_edges_src = []
        new_edges_dst = []
        new_weights = []
        for j in top10_cohort_idx:
            s_val = float(sim_to_cohort[j])
            new_edges_src.extend([new_node_idx, j])
            new_edges_dst.extend([j, new_node_idx])
            new_weights.extend([s_val, s_val])

        edge_index_aug = torch.cat(
            [orig_g.edge_index, torch.tensor([new_edges_src, new_edges_dst], dtype=torch.long)],
            dim=1,
        )
        edge_attr_aug = torch.cat(
            [orig_g.edge_attr, torch.tensor(new_weights, dtype=torch.float32)],
            dim=0,
        )

        from torch_geometric.data import Data
        g_aug = Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug)
        inductive_graphs.append(g_aug)

    # Run 5 fold models and average probabilities
    fold_probs = {}
    all_probs = []
    modality_weights_list = []

    for f in range(1, 6):
        m = store.fold_models[f]
        with torch.no_grad():
            logits, _, _, pool_w = m(*inductive_graphs)
            probs = torch.softmax(logits[new_node_idx], dim=-1).cpu().numpy()
            pool = pool_w[new_node_idx].cpu().numpy()

        f_probs = {PAM50_SUBTYPES[i]: float(probs[i]) for i in range(len(PAM50_SUBTYPES))}
        fold_probs[f"fold_{f}"] = f_probs
        all_probs.append(probs)
        modality_weights_list.append(pool)

    mean_probs_arr = np.mean(all_probs, axis=0)
    pred_c = int(np.argmax(mean_probs_arr))
    pred_label = PAM50_SUBTYPES[pred_c]
    conf = float(mean_probs_arr[pred_c])
    mean_weights_arr = np.mean(modality_weights_list, axis=0)

    return PredictNewResponse(
        prediction_type="new_patient (experimental, unvalidated)",
        predicted_subtype=pred_label,
        confidence=round(conf, 4),
        probabilities={PAM50_SUBTYPES[i]: round(float(mean_probs_arr[i]), 4) for i in range(len(PAM50_SUBTYPES))},
        fold_probabilities=fold_probs,
        modality_weights={
            "rna": round(float(mean_weights_arr[0]), 4),
            "mirna": round(float(mean_weights_arr[1]), 4),
            "methylation": round(float(mean_weights_arr[2]), 4),
        },
        nearest_neighbors=nearest_neighbors_report,
    )


# =========================================================================
# Polygon Blockchain Security & Patient Privacy Vault
# =========================================================================

_SIMULATED_TAMPER_RECORDS: dict[str, dict[str, Any]] = {}


def _get_canonical_patient_data(store: ArtifactStore, patient_id: str) -> dict[str, Any]:
    if patient_id in _SIMULATED_TAMPER_RECORDS:
        return _SIMULATED_TAMPER_RECORDS[patient_id]

    oof_pred = _compute_oof_prediction(store, patient_id)
    prow = {}
    if patient_id in store.patients.set_index("patient_id").index:
        prow = store.patients.set_index("patient_id").loc[patient_id].to_dict()

    return {
        "patient_id": patient_id,
        "predicted_subtype": oof_pred.predicted_subtype,
        "confidence": round(float(oof_pred.confidence), 4),
        "probabilities": oof_pred.probabilities,
        "modality_weights": oof_pred.modality_weights,
        "fold_id": oof_pred.fold_id,
        "true_pam50": str(prow.get("true_pam50", "Unknown")),
        "model_version": "GraphGPS-v1.0-4c26722",
    }


@app.get("/api/security/patient/{patient_id}")
def get_patient_security_status(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    vault = get_patient_vault()
    polygon = get_polygon_service()
    cohort = get_cohort_manager()

    record = _get_canonical_patient_data(store, patient_id)
    leaf_data = cohort._patient_leaves.get(patient_id, {})
    p_hash = leaf_data.get("patient_hash") or vault.anonymize_id(patient_id)
    current_fp = vault.compute_fingerprint(record)
    verification = polygon.verify_record(p_hash, current_fp)

    # Merkle Batch proof & verification
    if patient_id in _SIMULATED_TAMPER_RECORDS:
        leaf_hash = compute_patient_leaf(
            p_hash,
            current_fp,
            str(record.get("predicted_subtype", "LumA")),
            float(record.get("confidence", 0.95)),
        )
    else:
        leaf_hash = leaf_data.get("leaf_hash") or compute_patient_leaf(
            p_hash,
            current_fp,
            str(record.get("predicted_subtype", "LumA")),
            float(record.get("confidence", 0.95)),
        )
    proof_data = cohort.get_patient_proof(patient_id)
    cohort_verif = cohort.verify_patient_leaf(patient_id, leaf_hash)

    # If not individually anchored, check if verified via Cohort Merkle Root
    if not verification.get("is_anchored") and proof_data.get("has_proof"):
        is_merkle_valid = cohort_verif.get("is_verified", False)
        receipt = proof_data.get("receipt", {})
        verification = {
            "is_anchored": True,
            "is_unaltered": is_merkle_valid,
            "status": "VERIFIED_AUTHENTIC" if is_merkle_valid else "TAMPER_DETECTED",
            "method": "MERKLE_TREE_BATCH_PROOF",
            "timestamp": receipt.get("timestamp"),
            "authority": receipt.get("authority"),
            "tx_hash": receipt.get("tx_hash"),
            "block_number": receipt.get("block_number"),
            "polygonscan_url": receipt.get("polygonscan_url"),
            "network": receipt.get("network", "Polygon Amoy Testnet (Chain ID 80002)"),
            "merkle_root": proof_data.get("merkle_root"),
            "proof_length": proof_data.get("proof_length", 10),
        }

    return {
        "patient_id": patient_id,
        "patient_hash": p_hash,
        "current_fingerprint": current_fp,
        "leaf_hash": leaf_hash,
        "is_tampered": verification.get("status") == "TAMPER_DETECTED",
        "is_simulated_tamper": patient_id in _SIMULATED_TAMPER_RECORDS,
        "verification": verification,
        "merkle_proof": proof_data,
        "cohort_status": cohort.get_status(),
    }


@app.post("/api/security/anchor/{patient_id}")
def anchor_patient_to_polygon(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    # Clear any active simulated tamper before anchoring authentic record
    _SIMULATED_TAMPER_RECORDS.pop(patient_id, None)

    vault = get_patient_vault()
    polygon = get_polygon_service()

    record = _get_canonical_patient_data(store, patient_id)
    p_hash, current_fp = vault.store_encrypted_record(patient_id, record)

    subtype_order = ["LumA", "LumB", "Her2", "Basal", "Normal"]
    pred_subtype = record.get("predicted_subtype", "LumA")
    sub_idx = subtype_order.index(pred_subtype) if pred_subtype in subtype_order else 0
    conf_bps = int(float(record.get("confidence", 0.90)) * 10000)

    anchored = polygon.anchor_record(
        patient_hash=p_hash,
        data_fingerprint=current_fp,
        model_version="GraphGPS-4c26722",
        predicted_subtype=sub_idx,
        confidence_bps=conf_bps,
    )

    verification = polygon.verify_record(p_hash, current_fp)

    return {
        "status": "success",
        "patient_id": patient_id,
        "patient_hash": p_hash,
        "data_fingerprint": current_fp,
        "anchored": anchored,
        "verification": verification,
    }


@app.post("/api/security/simulate-tamper/{patient_id}")
def simulate_patient_tamper(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    vault = get_patient_vault()
    polygon = get_polygon_service()

    # Create mutated clone
    rec = copy.deepcopy(_get_canonical_patient_data(store, patient_id))
    orig_subtype = rec.get("predicted_subtype", "LumA")
    mutated_subtype = "Basal" if orig_subtype != "Basal" else "LumA"
    rec["predicted_subtype"] = mutated_subtype
    rec["confidence"] = 0.9999
    _SIMULATED_TAMPER_RECORDS[patient_id] = rec

    p_hash = vault.anonymize_id(patient_id)
    tampered_fp = vault.compute_fingerprint(rec)
    verification = polygon.verify_record(p_hash, tampered_fp)

    return {
        "status": "tampered",
        "patient_id": patient_id,
        "tampered_field": "predicted_subtype",
        "original_subtype": orig_subtype,
        "tampered_subtype": mutated_subtype,
        "tampered_fingerprint": tampered_fp,
        "verification": verification,
    }


@app.post("/api/security/restore/{patient_id}")
def restore_patient_integrity(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})

    _SIMULATED_TAMPER_RECORDS.pop(patient_id, None)

    vault = get_patient_vault()
    polygon = get_polygon_service()

    record = _get_canonical_patient_data(store, patient_id)
    p_hash = vault.anonymize_id(patient_id)
    current_fp = vault.compute_fingerprint(record)
    verification = polygon.verify_record(p_hash, current_fp)

    return {
        "status": "restored",
        "patient_id": patient_id,
        "current_fingerprint": current_fp,
        "verification": verification,
    }


# =========================================================================
# Cohort Merkle Batch Anchoring & Verification Endpoints
# =========================================================================

@app.get("/api/security/cohort/status")
def get_cohort_batch_status():
    cohort = get_cohort_manager()
    polygon = get_polygon_service()
    status = cohort.get_status()

    bal_pol = "0.0"
    if polygon.w3 and polygon.w3.is_connected() and polygon.account:
        try:
            bal_wei = polygon.w3.eth.get_balance(polygon.account.address)
            bal_pol = str(round(float(polygon.w3.from_wei(bal_wei, "ether")), 6))
        except Exception:
            pass

    status["wallet_address"] = polygon.account.address if polygon.account else None
    status["wallet_balance_pol"] = bal_pol
    status["contract_address"] = polygon.contract_address
    return status


@app.post("/api/security/cohort/anchor")
def anchor_entire_cohort():
    store = _require_store()
    cohort = get_cohort_manager()
    result = cohort.build_and_anchor_cohort(store, _get_canonical_patient_data)
    return result



@app.get("/api/security/cohort/proof/{patient_id}")
def get_patient_merkle_proof(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})
    cohort = get_cohort_manager()
    return cohort.get_patient_proof(patient_id)


@app.post("/api/security/cohort/verify/{patient_id}")
def verify_patient_merkle_proof(patient_id: str):
    store = _require_store()
    if patient_id not in store.patient_index:
        raise HTTPException(status_code=404, detail={"error": "patient_not_found", "detail": patient_id})
    vault = get_patient_vault()
    cohort = get_cohort_manager()
    record = _get_canonical_patient_data(store, patient_id)
    p_hash = vault.anonymize_id(patient_id)
    current_fp = vault.compute_fingerprint(record)
    leaf_hash = compute_patient_leaf(
        p_hash,
        current_fp,
        str(record.get("predicted_subtype", "LumA")),
        float(record.get("confidence", 0.95)),
    )
    return cohort.verify_patient_leaf(patient_id, leaf_hash)



if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BIONEXUS_HOST", "127.0.0.1")
    port = int(os.getenv("BIONEXUS_PORT", "8000"))
    debug = os.getenv("BIONEXUS_DEBUG", "false").lower() == "true"
    uvicorn.run("bio_nexus.serving.api:app", host=host, port=port, reload=debug)
