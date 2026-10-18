"""
Register the best model from a completed training run into the MLflow
Model Registry, in the "Staging" stage.

This is the missing link between "a training run happened" and "a model
is a deployable artifact." Without this step, a good MLflow run is just a
number in a dashboard — it never becomes something that can be promoted,
rolled back, or served.

MLOps contract: this reads run metadata from a file written by the
training/evaluation step (reports/experiment_info.json), NOT from a
hardcoded run_id. That file is what makes registration reproducible and
scriptable in CI rather than a manual "copy the run_id from the UI" step.

Usage:
    python -m bio_nexus.registry.register_model
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from bio_nexus.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = os.getenv("BIONEXUS_MODEL_NAME", "mosaic-subtype-classifier")
EXPERIMENT_INFO_PATH = Path("reports/experiment_info.json")


def _set_tracking_uri(info: dict | None = None) -> str:
    """Point MLflow at wherever tracking is hosted."""
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri and info and "tracking_uri" in info:
        uri = info["tracking_uri"]
    if not uri:
        db_path = Path("mlflow.db").resolve()
        uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    return uri


def load_experiment_info(path: Path = EXPERIMENT_INFO_PATH) -> dict:
    """Load {run_id, model_artifact_path, metrics} written by the training stage."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the training/evaluation stage first — it must "
            "write run_id + model_artifact_path here before registration."
        )
    with open(path) as f:
        return json.load(f)


def register_and_stage(model_name: str, info: dict) -> str:
    """Register the model, tag metrics, and transition it to Staging. Returns the new version."""
    model_uri = info.get("model_uri", f"runs:/{info['run_id']}/{info.get('model_artifact_path', 'model')}")
    result = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Staging",
            archive_existing_versions=False,
        )
    except Exception as stage_err:
        logger.warning(f"Stage transition warning: {stage_err}")

    # Set aliases and tags
    try:
        client.set_registered_model_alias(name=model_name, alias="staging", version=result.version)
    except Exception as alias_err:
        logger.debug(f"Alias setting warning: {alias_err}")

    tags = {
        "cohort": "TCGA-BRCA",
        "num_patients": "646",
        "architecture": "GraphGPS_CrossAttention",
        "run_id": str(info.get("run_id")),
    }
    metrics = info.get("metrics", {})
    for k, v in metrics.items():
        tags[f"metric_{k}"] = str(round(float(v), 4))

    for tag_k, tag_v in tags.items():
        try:
            client.set_model_version_tag(
                name=model_name,
                version=result.version,
                key=tag_k,
                value=tag_v,
            )
        except Exception as tag_err:
            logger.debug(f"Tag setting warning: {tag_err}")

    logger.info(f"Registered {model_name} v{result.version} -> Staging (tags={tags})")
    return result.version


def main() -> None:
    info = load_experiment_info()
    uri = _set_tracking_uri(info)
    logger.info(f"Using MLflow tracking URI: {uri}")
    register_and_stage(MODEL_NAME, info)


if __name__ == "__main__":
    main()
