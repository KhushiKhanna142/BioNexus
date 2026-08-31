"""
Promote the current Staging model to Production, archiving whatever was
Production before it.

This should ONLY run after tests/model/test_registered_model.py has
passed in CI — promotion is not "the training script finished," it's
"the training script finished AND the registered artifact passed sanity
checks (loads correctly, produces correctly-shaped output, beats the
XGBoost baseline)." See .github/workflows/ci.yml for the gating.

Usage:
    python -m bio_nexus.registry.promote_model
"""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from bio_nexus.registry.register_model import MODEL_NAME
from bio_nexus.utils.logging import get_logger

logger = get_logger(__name__)


def promote_latest_staging_to_production(model_name: str) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise OSError("MLFLOW_TRACKING_URI is not set.")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()

    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        raise RuntimeError(f"No model in Staging for '{model_name}' — nothing to promote.")
    new_version = staging_versions[0].version

    # Archive whatever is currently Production before promoting the new one,
    # so there's never a moment with two "Production" versions active.
    for version in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name, version=version.version, stage="Archived"
        )
        logger.info(f"Archived previous Production version {version.version}")

    client.transition_model_version_stage(name=model_name, version=new_version, stage="Production")
    logger.info(f"{model_name} v{new_version} promoted to Production")
    return new_version


if __name__ == "__main__":
    promote_latest_staging_to_production(MODEL_NAME)
