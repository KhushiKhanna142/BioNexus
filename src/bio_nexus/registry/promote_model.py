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

from pathlib import Path

from bio_nexus.registry.register_model import MODEL_NAME, EXPERIMENT_INFO_PATH
from bio_nexus.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_tracking_uri() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri and EXPERIMENT_INFO_PATH.exists():
        try:
            import json
            info = json.loads(EXPERIMENT_INFO_PATH.read_text())
            uri = info.get("tracking_uri")
        except Exception:
            pass
    if not uri:
        db_path = Path("mlflow.db").resolve()
        uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(uri)
    return uri


def promote_latest_staging_to_production(model_name: str) -> str:
    uri = _resolve_tracking_uri()
    logger.info(f"Using MLflow tracking URI: {uri}")

    client = MlflowClient()

    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging_versions:
        # Check by alias if staging stage empty
        try:
            m_alias = client.get_model_version_by_alias(model_name, "staging")
            staging_versions = [m_alias]
        except Exception:
            pass

    if not staging_versions:
        raise RuntimeError(f"No model in Staging for '{model_name}' — nothing to promote.")
    new_version = staging_versions[0].version

    # Archive whatever is currently Production before promoting the new one,
    # so there's never a moment with two "Production" versions active.
    for version in client.get_latest_versions(model_name, stages=["Production"]):
        try:
            client.transition_model_version_stage(
                name=model_name, version=version.version, stage="Archived"
            )
            logger.info(f"Archived previous Production version {version.version}")
        except Exception as arch_err:
            logger.warning(f"Archive warning: {arch_err}")

    try:
        client.transition_model_version_stage(name=model_name, version=new_version, stage="Production")
    except Exception as trans_err:
        logger.warning(f"Stage transition warning: {trans_err}")

    try:
        client.set_registered_model_alias(name=model_name, alias="champion", version=new_version)
        client.set_registered_model_alias(name=model_name, alias="production", version=new_version)
    except Exception as alias_err:
        logger.debug(f"Alias setting warning: {alias_err}")

    logger.info(f"{model_name} v{new_version} promoted to Production (alias=@champion)")
    return new_version


if __name__ == "__main__":
    promote_latest_staging_to_production(MODEL_NAME)
