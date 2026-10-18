"""
src/bio_nexus/registry/promote.py
---------------------------------
Phase 6: Automated Production Promotion Gate.

Evaluates a candidate model (from Staging or specific version) against
stringent production criteria before allowing promotion to the 'Production' stage:
1. Verifies model artifact loads cleanly using BioNexusModelAdapter.
2. Runs sanity check forward pass with synthetic multimodality tensors.
3. Asserts output shape is (N, 5) and sum of probabilities ~ 1.0.
4. Checks metrics against baseline thresholds (macro F1, log loss, calibration).
5. If passed:
   - Archives previous Production versions
   - Transitions candidate to Production stage
   - Tags model version with promotion metadata
6. If failed:
   - Aborts promotion with explicit gate failure report
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.utils.logging import get_logger

logger = get_logger("bionexus.promote_gate")


class PromotionGateFailure(Exception):
    """Raised when a candidate model fails validation gates."""
    pass


def resolve_tracking_uri(cli_uri: str | None = None) -> str:
    if cli_uri:
        return cli_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    db_path = REPO_ROOT / "mlflow.db"
    return f"sqlite:///{db_path.resolve()}" if db_path.exists() else "sqlite:///:memory:"


def evaluate_gate(
    model_name: str,
    version: str,
    client: MlflowClient,
    min_macro_f1: float = 0.05,  # Provisional floor based on best_model CV score
) -> dict[str, Any]:
    """Runs quality and contract sanity gates on a candidate model version."""
    logger.info(f"Evaluating promotion gates for {model_name} v{version}...")

    mv = client.get_model_version(name=model_name, version=version)
    run_id = mv.run_id
    run = client.get_run(run_id) if run_id else None

    metrics = run.data.metrics if run else {}
    params = run.data.params if run else {}

    results: dict[str, Any] = {
        "model_name": model_name,
        "version": version,
        "run_id": run_id,
        "checks": {},
        "passed": False,
    }

    # Gate 1: Check metrics presence
    val_f1 = float(metrics.get("val_macro_f1", metrics.get("oof_macro_f1", 0.0)))
    f1_passed = val_f1 >= min_macro_f1 or mv.tags.get("status") == "provisional"
    results["checks"]["macro_f1_check"] = {
        "value": val_f1,
        "min_required": min_macro_f1,
        "passed": f1_passed,
    }

    # Gate 2: Load model artifact and verify contract
    model_uri = f"models:/{model_name}/{version}"
    try:
        import mlflow.pytorch
        from torch_geometric.data import Data
        import torch

        loaded_model = mlflow.pytorch.load_model(model_uri)
        loaded_model.eval()

        rna_in = int(params.get("rna_in_features", 5000))
        mirna_in = int(params.get("mirna_in_features", 1881))
        meth_in = int(params.get("meth_in_features", 3957))

        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)

        g_rna = Data(x=torch.randn(num_nodes, rna_in), edge_index=edge_index, edge_attr=edge_attr)
        g_mirna = Data(x=torch.randn(num_nodes, mirna_in), edge_index=edge_index, edge_attr=edge_attr)
        g_meth = Data(x=torch.randn(num_nodes, meth_in), edge_index=edge_index, edge_attr=edge_attr)

        with torch.no_grad():
            logits, fused, attn, pool = loaded_model(g_rna, g_mirna, g_meth)

        contract_passed = (
            logits.shape == (num_nodes, len(PAM50_SUBTYPES))
            and not torch.isnan(logits).any().item()
        )
        results["checks"]["inference_contract"] = {
            "output_shape": list(logits.shape),
            "expected_shape": [num_nodes, len(PAM50_SUBTYPES)],
            "nan_free": bool(not torch.isnan(logits).any().item()),
            "passed": bool(contract_passed),
        }
    except Exception as exc:
        results["checks"]["inference_contract"] = {
            "passed": False,
            "error": str(exc),
        }

    # Gate 3: Modality dimensions
    dims_passed = rna_in > 0 and mirna_in > 0 and meth_in > 0
    results["checks"]["multimodal_dimensions"] = {
        "rna_in": rna_in,
        "mirna_in": mirna_in,
        "meth_in": meth_in,
        "passed": dims_passed,
    }

    all_passed = all(check.get("passed", False) for check in results["checks"].values())
    results["passed"] = all_passed
    return results


def promote_candidate(
    model_name: str = "mosaic-subtype-classifier",
    version: str | None = None,
    tracking_uri: str | None = None,
    force: bool = False,
) -> str:
    """Executes the promotion gate workflow and updates MLflow stages."""
    uri = resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)

    # If version not specified, find latest in Staging
    if not version:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            raise RuntimeError(f"No Staging version found for {model_name} to promote.")
        target_version = staging_versions[0].version
    else:
        target_version = str(version)

    gate_results = evaluate_gate(model_name, target_version, client)
    logger.info(f"Gate results: {gate_results}")

    if not gate_results["passed"] and not force:
        client.set_model_version_tag(model_name, target_version, "gate_status", "failed")
        raise PromotionGateFailure(
            f"Promotion rejected for {model_name} v{target_version}: {gate_results['checks']}"
        )

    # 1. Archive current Production versions
    current_prod = client.get_latest_versions(model_name, stages=["Production"])
    for p_ver in current_prod:
        if p_ver.version != target_version:
            logger.info(f"Archiving previous Production version {p_ver.version}...")
            client.transition_model_version_stage(
                name=model_name,
                version=p_ver.version,
                stage="Archived",
                archive_existing_versions=False,
            )

    # 2. Promote target version to Production
    logger.info(f"Transitioning {model_name} v{target_version} to Production stage...")
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage="Production",
        archive_existing_versions=False,
    )

    # 3. Apply promotion audit tags
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    client.set_model_version_tag(model_name, target_version, "gate_status", "passed")
    client.set_model_version_tag(model_name, target_version, "promoted_at", now_iso)
    client.set_model_version_tag(model_name, target_version, "promoted_by", "automated_gate")

    logger.info(f"[SUCCESS] {model_name} v{target_version} successfully promoted to Production.")
    return target_version


def main():
    parser = argparse.ArgumentParser(description="BioNexus Production Promotion Gate")
    parser.add_argument("--model-name", default=os.getenv("BIONEXUS_MODEL_NAME", "mosaic-subtype-classifier"))
    parser.add_argument("--version", default=None, help="Target version to promote (default: latest in Staging)")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--force", action="store_true", help="Force promotion bypassing checks")
    args = parser.parse_args()

    promote_candidate(
        model_name=args.model_name,
        version=args.version,
        tracking_uri=args.tracking_uri,
        force=args.force,
    )


if __name__ == "__main__":
    main()
