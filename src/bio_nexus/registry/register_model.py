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

MODEL_NAME = "bionexus-subtype-classifier"
EXPERIMENT_INFO_PATH = Path("reports/experiment_info.json")


def _set_tracking_uri() -> None:
    """Point MLflow at wherever tracking is hosted.

    Reads MLFLOW_TRACKING_URI from the environment. For local dev this is
    the self-hosted server from docker-compose.yml (http://localhost:5000).
    For a free hosted option that survives your machine being off and is
    easy to link in a README/resume, DagsHub's MLflow-compatible tracking
    server is a solid choice for a student project — set
    MLFLOW_TRACKING_URI and MLFLOW_TRACKING_USERNAME/PASSWORD (or a token)
    as environment variables / GitHub Actions secrets, never hardcoded here.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise OSError(
            "MLFLOW_TRACKING_URI is not set. Point it at your local MLflow "
            "server or a hosted tracking server (e.g. DagsHub)."
        )
    mlflow.set_tracking_uri(uri)


def load_experiment_info(path: Path = EXPERIMENT_INFO_PATH) -> dict:
    """Load {run_id, model_artifact_path, metrics} written by the eval stage."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the evaluation stage first — it must "
            "write run_id + model_artifact_path here before registration."
        )
    with open(path) as f:
        return json.load(f)


def register_and_stage(model_name: str, info: dict) -> str:
    """Register the model and transition it to Staging. Returns the new version."""
    model_uri = f"runs:/{info['run_id']}/{info['model_artifact_path']}"
    result = mlflow.register_model(model_uri, model_name)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info(f"Registered {model_name} v{result.version} -> Staging")
    return result.version


def main() -> None:
    _set_tracking_uri()
    info = load_experiment_info()
    register_and_stage(MODEL_NAME, info)


if __name__ == "__main__":
    main()
