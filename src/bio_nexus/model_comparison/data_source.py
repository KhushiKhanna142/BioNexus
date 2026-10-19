"""
Loading + validation of comparison results.

Two interchangeable sources:
  * JsonFileSource  - default; hot-reloads when the file changes.
  * MLflowSource    - reads final runs straight from MLflow (teammate's tracking server).

Both return a validated `Dataset`. Nothing else in the package knows where data came from.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .registry import METRIC_BY_ID, MINORITY_CLASSES, PAM50_CLASSES


class DataValidationError(ValueError):
    pass


@dataclass
class ModelResult:
    id: str
    name: str
    family: str
    metrics: dict[str, Any]
    per_class_f1: dict[str, float | None]


@dataclass
class Dataset:
    models: list[ModelResult]
    data_status: str = "placeholder"   # "placeholder" | "measured"
    notes: str = ""
    by_id: dict[str, ModelResult] = field(init=False)

    def __post_init__(self):
        self.by_id = {m.id: m for m in self.models}


class ResultsSource(Protocol):
    def load(self) -> Dataset: ...


# ------------------------------------------------------------------ parsing
def _check_number(model_id: str, key: str, v: Any, lo: float | None = None) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise DataValidationError(f"{model_id}.{key}: expected a finite number or null, got {v!r}")
    if lo is not None and v < lo:
        raise DataValidationError(f"{model_id}.{key}: must be >= {lo}, got {v}")


def parse_dataset(raw: dict) -> Dataset:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list) or not raw["models"]:
        raise DataValidationError("results must be an object with a non-empty 'models' list")

    seen: set[str] = set()
    models: list[ModelResult] = []
    for item in raw["models"]:
        try:
            mid, name = item["id"], item["name"]
        except (KeyError, TypeError):
            raise DataValidationError("each model needs 'id' and 'name'") from None
        if mid in seen:
            raise DataValidationError(f"duplicate model id: {mid}")
        seen.add(mid)

        metrics = dict(item.get("metrics") or {})
        per_class = dict(item.get("per_class_f1") or {})

        unknown = set(metrics) - set(METRIC_BY_ID)
        if unknown:
            raise DataValidationError(f"{mid}: unknown metric(s) {sorted(unknown)}")
        bad_cls = set(per_class) - set(PAM50_CLASSES)
        if bad_cls:
            raise DataValidationError(f"{mid}: unknown PAM50 class(es) {sorted(bad_cls)}")

        for key, v in metrics.items():
            if v is None:
                continue
            if METRIC_BY_ID[key].kind == "boolean":
                if not isinstance(v, bool):
                    raise DataValidationError(f"{mid}.{key}: expected true/false/null, got {v!r}")
            else:
                _check_number(mid, key, v, lo=0)
        for cls, v in per_class.items():
            if v is not None:
                _check_number(mid, f"per_class_f1.{cls}", v, lo=0)
                if v > 1:
                    raise DataValidationError(f"{mid}.per_class_f1.{cls}: F1 must be in [0, 1]")

        # Derive rare-subtype F1 from per-class scores so the two can never disagree.
        if metrics.get("minority_f1") is None:
            vals = [per_class.get(c) for c in MINORITY_CLASSES]
            if all(v is not None for v in vals):
                metrics["minority_f1"] = sum(vals) / len(vals)

        models.append(ModelResult(mid, name, item.get("family", ""), metrics, per_class))

    return Dataset(models, raw.get("data_status", "placeholder"), raw.get("notes", ""))


# ------------------------------------------------------------------ sources
class JsonFileSource:
    """Reads a JSON file; re-parses automatically when its mtime changes."""

    def __init__(self, path: str | os.PathLike | None = None):
        default = Path(__file__).parent / "data" / "results.json"
        self.path = Path(path or os.getenv("MODEL_COMPARISON_RESULTS", default))
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._cached: Dataset | None = None

    def load(self) -> Dataset:
        with self._lock:
            mtime = self.path.stat().st_mtime
            if self._cached is None or mtime != self._mtime:
                with open(self.path, encoding="utf-8") as f:
                    self._cached = parse_dataset(json.load(f))
                self._mtime = mtime
            return self._cached

    def invalidate(self) -> None:
        with self._lock:
            self._cached, self._mtime = None, None


class MLflowSource:
    """
    Pulls the newest FINISHED run per model from an MLflow experiment.

    Conventions the MLOps side needs to follow when logging a final run:
      tags:    model_id=<id>, stage=final, display_name=<nice name>, family=<family>
      metrics: names equal to the metric ids in registry.py
               (accuracy, macro_f1, auc_roc, cross_cohort_drop, noise_robustness,
                cosmic_overlap_pct, train_time_min, inference_ms, params_m)
               booleans as 1.0/0.0 (attention_explanations, causal_ablation, gpu_required)
               per-class F1 as f1_LumA, f1_LumB, f1_Her2, f1_Basal, f1_Normal
    """

    def __init__(self, experiment_name: str, tracking_uri: str | None = None,
                 stage_tag: str = "final", ttl_seconds: int = 60):
        self.experiment_name, self.tracking_uri = experiment_name, tracking_uri
        self.stage_tag, self.ttl = stage_tag, ttl_seconds
        self._cached: Dataset | None = None
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def load(self) -> Dataset:
        import time
        with self._lock:
            if self._cached is None or time.time() - self._loaded_at > self.ttl:
                self._cached = self._fetch()
                self._loaded_at = time.time()
            return self._cached

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None

    def _fetch(self) -> Dataset:
        from mlflow.tracking import MlflowClient  # lazy: only needed for this source

        client = MlflowClient(tracking_uri=self.tracking_uri)
        exp = client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            raise DataValidationError(f"MLflow experiment not found: {self.experiment_name}")
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string=f"tags.stage = '{self.stage_tag}' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
        )
        latest: dict[str, Any] = {}
        for run in runs:                      # newest first -> first seen wins
            mid = run.data.tags.get("model_id")
            if mid and mid not in latest:
                latest[mid] = run

        models = []
        for mid, run in latest.items():
            m = run.data.metrics
            metrics: dict[str, Any] = {}
            for key, mdef in METRIC_BY_ID.items():
                if key in m:
                    metrics[key] = bool(m[key]) if mdef.kind == "boolean" else float(m[key])
            per_class = {c: float(m[f"f1_{c}"]) for c in PAM50_CLASSES if f"f1_{c}" in m}
            models.append({
                "id": mid,
                "name": run.data.tags.get("display_name", mid),
                "family": run.data.tags.get("family", ""),
                "metrics": metrics,
                "per_class_f1": per_class,
            })
        return parse_dataset({"data_status": "measured", "notes": "Loaded from MLflow.", "models": models})


def default_source() -> ResultsSource:
    """MODEL_COMPARISON_SOURCE=mlflow (+ MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI) or json (default)."""
    if os.getenv("MODEL_COMPARISON_SOURCE", "json").lower() == "mlflow":
        return MLflowSource(os.environ["MLFLOW_EXPERIMENT"], os.getenv("MLFLOW_TRACKING_URI"))
    return JsonFileSource()
