"""
FastAPI router for the Model Comparison dashboard feature.

Mount it in the existing app (see app_example.py):
    app.include_router(create_router(default_source()))

All responses are plain JSON built in analysis.py. The frontend can either call
GET /dashboard once (everything the page needs) or the granular endpoints.
"""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, Header, HTTPException, Query

from . import analysis
from .data_source import DataValidationError, ResultsSource
from .registry import CATEGORIES, FOCUS_MODEL_ID, METRICS, RADAR_AXES

MODELS_Q = Query(None, description="Comma-separated model ids to compare, e.g. mogonet,xgboost. "
                                   "The focus model is always included. Omit for all models.")
FOCUS_Q = Query(FOCUS_MODEL_ID, description="Model the dashboard highlights.")


def _ids(models: str | None) -> list[str] | None:
    ids = [s.strip() for s in models.split(",") if s.strip()] if models else None
    return ids or None


def create_router(source: ResultsSource) -> APIRouter:
    router = APIRouter(prefix="/api/v1/model-comparison", tags=["model-comparison"])

    def _dataset():
        try:
            return source.load()
        except FileNotFoundError:
            raise HTTPException(503, "comparison results file not found")
        except DataValidationError as e:
            raise HTTPException(500, f"invalid comparison results: {e}")
        except Exception as e:  # e.g. MLflow unreachable
            raise HTTPException(503, f"comparison results unavailable: {e}")

    def _run(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except analysis.ComparisonError as e:
            raise HTTPException(404 if "not found" in str(e) or "unknown model" in str(e) else 422, str(e))

    @router.get("/meta")
    def meta():
        """Static definitions: models, metrics (with tooltips), categories, radar axes."""
        ds = _dataset()
        return {
            "data_status": ds.data_status, "notes": ds.notes, "focus_model_id": FOCUS_MODEL_ID,
            "models": [{"id": m.id, "name": m.name, "family": m.family} for m in ds.models],
            "categories": CATEGORIES,
            "metrics": [{"id": m.id, "label": m.label, "category": m.category, "direction": m.direction,
                         "unit": m.unit, "kind": m.kind, "description": m.description} for m in METRICS],
            "radar_axes": [{"id": a["id"], "label": a["label"]} for a in RADAR_AXES],
        }

    @router.get("/dashboard")
    def dashboard(models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """Everything the page needs in one call: summary, leaderboard, insights, radar, per-class."""
        return _run(analysis.build_dashboard, _dataset(), _ids(models), focus)

    @router.get("/leaderboard")
    def leaderboard(category: str | None = Query(None, description="performance | robustness | interpretability | cost"),
                    models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """Per-metric ranking table (values, ranks, winners)."""
        ds = _dataset()
        sel = _run(analysis.select_models, ds, _ids(models), focus)
        return {"metrics": _run(analysis.leaderboard, sel, category)}

    @router.get("/summary")
    def summary(models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """'Leads on X of Y metrics' headline + per-category breakdown."""
        sel = _run(analysis.select_models, _dataset(), _ids(models), focus)
        return analysis.summarize(analysis.leaderboard(sel), sel, focus)

    @router.get("/insights")
    def insights(models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """Where the focus model excels vs where others win, with reasons."""
        sel = _run(analysis.select_models, _dataset(), _ids(models), focus)
        return analysis.insights(analysis.leaderboard(sel), sel, focus)

    @router.get("/radar")
    def radar(models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """0-100 normalised scores per axis for the radar chart."""
        sel = _run(analysis.select_models, _dataset(), _ids(models), focus)
        return analysis.radar(sel)

    @router.get("/per-class")
    def per_class(models: str | None = MODELS_Q, focus: str = FOCUS_Q):
        """Per-PAM50-subtype F1 matrix for the heatmap."""
        sel = _run(analysis.select_models, _dataset(), _ids(models), focus)
        return analysis.per_class(sel, focus)

    @router.post("/reload")
    def reload(x_admin_token: str | None = Header(None)):
        """Force re-read of results (e.g. after a new MLflow run). Requires MODEL_COMPARISON_ADMIN_TOKEN."""
        expected = os.getenv("MODEL_COMPARISON_ADMIN_TOKEN")
        if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
            raise HTTPException(403, "forbidden")
        if hasattr(source, "invalidate"):
            source.invalidate()
        ds = _dataset()
        return {"status": "reloaded", "models": len(ds.models), "data_status": ds.data_status}

    return router
