"""
scripts/run_mlflow_pipeline.py
------------------------------
End-to-end MLflow execution pipeline:
1. Initializes tracking backend (local SQLite or MLFLOW_TRACKING_URI).
2. Runs GraphGPS multi-omics model cross-validation or fast smoke-test.
3. Logs hyperparameters, per-fold metrics, aggregate macro-F1/accuracy, and artifacts.
4. Logs PyTorch model to MLflow artifact repository.
5. Registers model in MLflow Model Registry and transitions to 'Staging'.
6. Optionally promotes best model to 'Production'.
7. Displays summary and command to launch MLflow UI.

Usage:
    python scripts/run_mlflow_pipeline.py [--smoke-test] [--promote] [--epochs 50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add src/ to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow
import numpy as np
import torch

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.registry.promote_model import promote_latest_staging_to_production
from bio_nexus.registry.register_model import MODEL_NAME, register_and_stage
from bio_nexus.train import (
    TrainConfig,
    _get_mlflow_tracking_uri,
    _maybe_finish_mlflow,
    _maybe_start_mlflow,
    load_data,
    set_seed,
    train_one_fold,
)
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(description="BioNexus MLflow Training & Registry Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 fast epochs on 2 folds for verification")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--promote", action="store_true", help="Automatically promote Staging model to Production")
    parser.add_argument("--tracking-uri", type=str, default=None, help="Custom MLflow tracking URI")
    parser.add_argument("--experiment", type=str, default="bionexus-cancer-subtype", help="MLflow experiment name")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]

    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT"] = args.experiment

    tracking_uri = _get_mlflow_tracking_uri(base_dir)
    print(f"\n=======================================================")
    print(f"   BioNexus MLflow Pipeline")
    print(f"   Tracking URI:    {tracking_uri}")
    print(f"   Experiment:      {args.experiment}")
    print(f"   Model Name:      {MODEL_NAME}")
    print(f"=======================================================\n")

    cfg = TrainConfig(
        n_folds=2 if args.smoke_test else args.n_folds,
        max_epochs=2 if args.smoke_test else args.epochs,
        lr=args.lr,
        seed=42,
    )
    set_seed(cfg.seed)

    print("1. Loading 646 TCGA-BRCA patient graphs & multi-omics matrices...")
    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(base_dir)
    print(f"   Loaded cohort with {len(y)} patients across 5 PAM50 subtypes.")

    extra_params = {
        "rna_in_features": int(rna_graph.x.shape[1]),
        "mirna_in_features": int(mirna_graph.x.shape[1]),
        "meth_in_features": int(meth_graph.x.shape[1]),
        "num_patients": int(len(y)),
        "is_smoke_test": args.smoke_test,
    }

    print("\n2. Initializing MLflow run...")
    run_id = _maybe_start_mlflow(cfg, extra_params=extra_params, base_dir=base_dir)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_metrics = []
    best_fold_model = None
    best_fold_f1 = -1.0
    ckpt_dir = base_dir / "data" / "processed" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n3. Executing {cfg.n_folds}-fold cross-validation training...")
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n--- Training Fold {fold_num}/{cfg.n_folds} ---")
        model, metrics = train_one_fold(
            rna_graph, mirna_graph, meth_graph, y, train_idx, val_idx, cfg, fold_num
        )
        fold_metrics.append(metrics)
        torch.save(model.state_dict(), ckpt_dir / f"fold_{fold_num}.pt")

        if run_id:
            try:
                mlflow.log_metrics(
                    {
                        f"fold_{fold_num}_accuracy": float(metrics["accuracy"]),
                        f"fold_{fold_num}_macro_f1": float(metrics["macro_f1"]),
                        f"fold_{fold_num}_weighted_f1": float(metrics["weighted_f1"]),
                    },
                    step=fold_num,
                )
            except Exception as e:
                print(f"[mlflow] Warning logging fold metrics: {e}")

        if metrics["macro_f1"] > best_fold_f1:
            best_fold_f1 = metrics["macro_f1"]
            best_fold_model = model

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    wf1s = [m["weighted_f1"] for m in fold_metrics]

    print(f"\n=======================================================")
    print(f"   Cross-Validation Results ({cfg.n_folds} folds)")
    print(f"   Mean Accuracy:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"   Mean Macro F1:     {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"   Mean Weighted F1:  {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}")
    print(f"=======================================================\n")

    torch.save(best_fold_model.state_dict(), ckpt_dir / "best_model.pt")

    results_dir = base_dir / "data" / "processed" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fused_model": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "weighted_f1_mean": float(np.mean(wf1s)),
            "weighted_f1_std": float(np.std(wf1s)),
            "confusion_matrix": {
                "labels": PAM50_SUBTYPES,
                "matrix": fold_metrics[int(np.argmax(f1s))]["confusion_matrix"].tolist(),
            },
        },
        "n_patients": int(len(y)),
        "n_folds": cfg.n_folds,
    }
    (results_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    print("4. Finalizing MLflow run and logging PyTorch model artifact...")
    _maybe_finish_mlflow(run_id, payload, base_dir, best_model=best_fold_model)

    if run_id:
        print("\n5. Registering model in MLflow Model Registry...")
        info_path = base_dir / "reports" / "experiment_info.json"
        if info_path.exists():
            exp_info = json.loads(info_path.read_text())
            version = register_and_stage(MODEL_NAME, exp_info)
            print(f"   Successfully registered '{MODEL_NAME}' version v{version} -> Staging")

            if args.promote:
                print("\n6. Promoting model to Production...")
                prod_v = promote_latest_staging_to_production(MODEL_NAME)
                print(f"   Successfully promoted '{MODEL_NAME}' v{prod_v} -> Production (@champion)")

    print(f"\n=======================================================")
    print(f"   MLflow Pipeline Complete!")
    print(f"   Run ID:          {run_id}")
    print(f"   To view MLflow UI, run:")
    print(f"   mlflow ui --backend-store-uri {tracking_uri}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
