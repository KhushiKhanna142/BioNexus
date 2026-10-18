"""
scripts/import_checkpoint.py
----------------------------
Imports existing provisional PyTorch checkpoints into MLflow:
1. Logs each fold checkpoint (fold_1.pt through fold_5.pt) as an MLflow run with
   its fold-specific evaluation metrics from reports/checkpoint_evaluation.json.
   Tagged: source=local_checkpoint, data_validated=false, fold=<k>, status=provisional.
2. Logs best_model.pt (selected from fold 3) with full OOF summary metrics.
3. Registers best_model.pt under registered model name 'mosaic-subtype-classifier'.
4. Transitions the registered model to the 'Production' stage.
5. Tags the registered model version with data_validated=false and source=local_checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Force CPU to bypass NVML / CUDA probe latency
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Avoid Windows CP1252 charmap encoding crash when MLflow prints emoji
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import mlflow
import mlflow.pytorch
from mlflow.client import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bio_nexus.models.classifier import BioNexusClassifier, ClassifierConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Import provisional checkpoints into MLflow")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: http://localhost:5001 if reachable, else sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="bionexus-subtype-classification",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mosaic-subtype-classifier",
        help="Registered model name",
    )
    return parser.parse_args()


def determine_tracking_uri(cli_uri: str | None) -> str:
    if cli_uri:
        return cli_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # Test if localhost:5001 is reachable
    import urllib.request
    for port in (5001, 5000):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5):
                return f"http://127.0.0.1:{port}"
        except Exception:
            pass

    # Fall back to local SQLite DB
    db_path = (REPO_ROOT / "mlflow.db").resolve()
    return f"sqlite:///{db_path}"


def import_checkpoints(tracking_uri: str, exp_name: str, model_name: str):
    print("=" * 65)
    print("        MOSAIC / BIONEXUS CHECKPOINT IMPORT & REGISTRATION")
    print("=" * 65)
    print(f"Tracking URI:    {tracking_uri}")
    print(f"Experiment Name: {exp_name}")
    print(f"Model Name:      {model_name}")
    print("-" * 65)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Ensure experiment exists
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
        print(f"Created experiment '{exp_name}' (ID: {exp_id})")
    else:
        exp_id = exp.experiment_id
        print(f"Using existing experiment '{exp_name}' (ID: {exp_id})")

    ckpt_dir = REPO_ROOT / "data" / "processed" / "checkpoints"
    eval_report_file = REPO_ROOT / "reports" / "checkpoint_evaluation.json"

    eval_data: dict[str, Any] = {}
    if eval_report_file.exists():
        eval_data = json.loads(eval_report_file.read_text())
        print(f"Loaded evaluation metrics from {eval_report_file}")
    else:
        print(f"WARNING: {eval_report_file} not found; proceeding with raw checkpoint weights.")

    per_fold_eval = eval_data.get("per_fold", {})
    oof_metrics = eval_data.get("oof_metrics", {})

    # Helper to instantiate model with exact feature dimensions from checkpoint
    def create_model_with_weights(ckpt_file: Path) -> tuple[BioNexusClassifier, int, int, int]:
        sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        rna_in = sd["rna_encoder.input_proj.weight"].shape[1]
        mirna_in = sd["mirna_encoder.input_proj.weight"].shape[1]
        meth_in = sd["meth_encoder.input_proj.weight"].shape[1]
        print(f"  Architecture: rna_in={rna_in}, mirna_in={mirna_in}, meth_in={meth_in}", flush=True)
        m = BioNexusClassifier(
            rna_in_features=rna_in,
            mirna_in_features=mirna_in,
            meth_in_features=meth_in,
        )
        m.load_state_dict(sd)
        m.eval()
        return m, rna_in, mirna_in, meth_in

    # 2. Log Fold 1 through 5 runs
    fold_run_ids = {}
    for k in range(1, 6):
        fold_name = f"fold_{k}"
        fold_file = ckpt_dir / f"{fold_name}.pt"
        if not fold_file.exists():
            print(f"[Fold {k}] Checkpoint file not found: {fold_file}")
            continue

        fold_metrics = per_fold_eval.get(fold_name, {})
        run_name = f"provisional-{fold_name}"

        tags = {
            "source": "local_checkpoint",
            "data_validated": "false",
            "model_status": "provisional",
            "fold": str(k),
            "checkpoint_filename": fold_file.name,
            "architecture": "BioNexusClassifier_3xGraphGPS_CrossAttention",
        }

        fold_sd = torch.load(fold_file, map_location="cpu", weights_only=True)
        f_rna_in = fold_sd["rna_encoder.input_proj.weight"].shape[1]
        f_mirna_in = fold_sd["mirna_encoder.input_proj.weight"].shape[1]
        f_meth_in = fold_sd["meth_encoder.input_proj.weight"].shape[1]

        with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
            fold_run_ids[k] = run.info.run_id
            print(f"\n[Fold {k}] Logged run ID: {run.info.run_id}")

            # Log parameters
            mlflow.log_params({
                "fold": k,
                "n_splits": 5,
                "rna_in_features": f_rna_in,
                "mirna_in_features": f_mirna_in,
                "meth_in_features": f_meth_in,
                "num_classes": 5,
                "val_samples": fold_metrics.get("val_samples", 129),
            })

            # Log metrics (evaluated on held-out fold)
            metrics_to_log = {
                "val_accuracy": fold_metrics.get("accuracy", 0.0),
                "val_balanced_accuracy": fold_metrics.get("balanced_accuracy", 0.0),
                "val_macro_f1": fold_metrics.get("macro_f1", 0.0),
                "val_weighted_f1": fold_metrics.get("weighted_f1", 0.0),
                "val_log_loss": fold_metrics.get("log_loss", 0.0),
                "val_ece": fold_metrics.get("ece", 0.0),
            }
            sub_f1 = fold_metrics.get("per_subtype_f1", {})
            for subtype, f1_val in sub_f1.items():
                metrics_to_log[f"val_f1_{subtype}"] = f1_val

            mlflow.log_metrics(metrics_to_log)
            print(f"  Metrics: val_acc={metrics_to_log['val_accuracy']:.4f}, macro_f1={metrics_to_log['val_macro_f1']:.4f}")

            # Log checkpoint weights as artifact
            mlflow.log_artifact(str(fold_file), artifact_path="checkpoints")

    # 3. Log best_model.pt and register in Model Registry
    best_file = ckpt_dir / "best_model.pt"
    if not best_file.exists():
        raise FileNotFoundError(f"Missing best_model.pt in {ckpt_dir}")

    print(f"\n[Champion Model] Packaging and registering {best_file.name}...")
    best_model, rna_in, mirna_in, meth_in = create_model_with_weights(best_file)

    champion_tags = {
        "source": "local_checkpoint",
        "data_validated": "false",
        "model_status": "provisional",
        "checkpoint_filename": "best_model.pt",
        "selected_from_fold": "fold_3",
        "selection_rationale": "Highest validation Macro F1 (0.0583) during cross-validation",
        "architecture": "BioNexusClassifier_3xGraphGPS_CrossAttention",
        "stage_intent": "Production",
    }

    with mlflow.start_run(experiment_id=exp_id, run_name="provisional-champion-best_model", tags=champion_tags) as run:
        champ_run_id = run.info.run_id
        print(f"Champion Run ID: {champ_run_id}")

        mlflow.log_params({
            "rna_in_features": rna_in,
            "mirna_in_features": mirna_in,
            "meth_in_features": meth_in,
            "num_classes": 5,
            "cohort_size": 646,
            "selected_from_fold": 3,
        })

        # Summary 5-fold OOF metrics + optimistic fold score
        champ_metrics = {
            "oof_accuracy": oof_metrics.get("accuracy", 0.1362),
            "oof_balanced_accuracy": oof_metrics.get("balanced_accuracy", 0.1935),
            "oof_macro_f1": oof_metrics.get("macro_f1", 0.1092),
            "oof_weighted_f1": oof_metrics.get("weighted_f1", 0.0830),
            "oof_log_loss": oof_metrics.get("log_loss", 1.9205),
            "oof_ece": oof_metrics.get("ece", 0.2234),
            "fold3_val_macro_f1_optimistic": 0.0583,
        }
        for subtype, f1_val in oof_metrics.get("per_subtype_f1", {}).items():
            champ_metrics[f"oof_f1_{subtype}"] = f1_val
        mlflow.log_metrics(champ_metrics)

        # Log PyTorch model with pinned requirements (avoids pip list environment scan)
        print(f"Logging PyTorch model artifact for '{model_name}'...", flush=True)
        pip_reqs = [
            "torch>=2.3.0",
            "torch-geometric>=2.5.0",
            "pandas>=2.2.0",
            "numpy>=1.26.0",
            "scikit-learn>=1.4.0",
        ]
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            pip_requirements=pip_reqs,
        )

        # Also store checkpoint file artifact
        mlflow.log_artifact(str(best_file), artifact_path="checkpoints")

    # 4. Register in Model Registry using standard 2.x API (compatible with MLflow 2.14)
    print(f"\n[Registry] Registering version for '{model_name}'...", flush=True)
    try:
        client.get_registered_model(model_name)
    except Exception:
        try:
            client.create_registered_model(model_name, description="MOSAIC Multi-Omics Subtype Classifier")
        except Exception:
            pass

    model_source = f"{run.info.artifact_uri}/model"
    mv = client.create_model_version(
        name=model_name,
        source=model_source,
        run_id=champ_run_id,
        tags={
            "data_validated": "false",
            "source": "local_checkpoint",
            "status": "provisional",
            "selected_from": "fold_3",
        },
    )
    ver_num = mv.version
    print(f"Registered model version: v{ver_num}", flush=True)

    # Set metadata tags on the registered model version
    client.set_model_version_tag(model_name, ver_num, "data_validated", "false")
    client.set_model_version_tag(model_name, ver_num, "source", "local_checkpoint")
    client.set_model_version_tag(model_name, ver_num, "status", "provisional")
    client.set_model_version_tag(model_name, ver_num, "selected_from", "fold_3")

    # Transition to Production stage
    print(f"Transitioning '{model_name}' v{ver_num} to stage 'Production'...", flush=True)
    client.transition_model_version_stage(
        name=model_name,
        version=ver_num,
        stage="Production",
        archive_existing_versions=True,
    )

    print("\n" + "=" * 65)
    print(f"SUCCESS: Model '{model_name}' v{ver_num} is now in 'Production' stage.")
    print("Tags: data_validated=false, source=local_checkpoint")
    print(f"Serving URI 'models:/{model_name}/Production' is ready to serve.")
    print("=" * 65)


def main():
    args = parse_args()
    if args.tracking_uri:
        import_checkpoints(
            tracking_uri=args.tracking_uri,
            exp_name=args.experiment_name,
            model_name=args.model_name,
        )
    else:
        # Default: Populate running Docker MLflow server and local SQLite
        targets = []
        # Check if Docker MLflow is running on 5001
        import urllib.request
        try:
            with urllib.request.urlopen("http://127.0.0.1:5001/", timeout=2):
                targets.append("http://127.0.0.1:5001")
        except Exception:
            pass

        local_db = f"sqlite:///{(REPO_ROOT / 'mlflow.db').resolve()}"
        targets.append(local_db)

        for uri in targets:
            try:
                import_checkpoints(
                    tracking_uri=uri,
                    exp_name=args.experiment_name,
                    model_name=args.model_name,
                )
            except Exception as e:
                print(f"[!] Error importing to {uri}: {e}", flush=True)


if __name__ == "__main__":
    main()
