"""
bio_nexus.train
--------------------
Training pipeline entrypoint. Trains BioNexusClassifier using stratified
K-fold cross-validation over the 646-patient cohort.

TRANSDUCTIVE SETTING (important design note for your methods section):
All 646 patients exist as nodes in one shared graph throughout training.
Cross-validation works by masking which nodes' LABELS are visible during
training vs evaluation - not by rebuilding separate graphs per fold. This
is the standard approach for GNN cross-validation (node features, including
those of validation-fold patients, are always visible during message-passing;
only their labels are hidden during the loss computation). This also means
the earlier methylation variance-filtering (unsupervised, uses no label
information) is standard transductive preprocessing, not label leakage -
worth stating explicitly as a design decision in your paper rather than
treating it as something that needed "fixing".

Handles class imbalance (PAM50 subtypes: LumA ~50%, Normal ~4.5%) via
class-weighted loss, and guards against overfitting (4.68M params vs
646 patients) via weight decay + early stopping on validation macro-F1.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier


@dataclass
class TrainConfig:
    n_folds: int = 5
    max_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4  # L2 regularization - important given param/patient ratio
    early_stop_patience: int = 20  # epochs without val macro-F1 improvement before stopping
    seed: int = 42


def _get_mlflow_tracking_uri(base_dir: Path | None = None) -> str | None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Optional default local SQLite tracking URI for standalone experiments
    if os.getenv("ENABLE_LOCAL_MLFLOW", "1").lower() in ("1", "true", "yes"):
        if base_dir:
            db_path = (base_dir / "mlflow.db").resolve()
            return f"sqlite:///{db_path}"
        return "sqlite:///mlflow.db"
    return None


def _maybe_start_mlflow(cfg: TrainConfig, extra_params: dict | None = None, base_dir: Path | None = None) -> str | None:
    uri = _get_mlflow_tracking_uri(base_dir)
    if not uri:
        print("[mlflow] MLFLOW_TRACKING_URI unset — skipping tracking.")
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        exp_name = os.getenv("MLFLOW_EXPERIMENT", "bionexus-cancer-subtype")
        mlflow.set_experiment(exp_name)
        run = mlflow.start_run()
        params = {
            "n_folds": cfg.n_folds,
            "max_epochs": cfg.max_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "early_stop_patience": cfg.early_stop_patience,
            "seed": cfg.seed,
            "architecture": "graphgps_cross_attention",
            "cohort": "TCGA-BRCA",
            "num_subtypes": len(PAM50_SUBTYPES),
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)
        mlflow.set_tags({
            "project": "bio_nexus",
            "task": "pam50_subtyping",
            "model_family": "GraphGPS",
            "framework": "PyTorch Geometric",
        })
        print(f"[mlflow] Started tracking run: id={run.info.run_id}, uri={uri}, experiment={exp_name}")
        return run.info.run_id
    except Exception as exc:
        print(f"[mlflow] failed to start: {exc}")
        return None


def _maybe_finish_mlflow(
    run_id: str | None,
    payload: dict,
    base_dir: Path,
    best_model: nn.Module | None = None,
) -> None:
    if not run_id:
        return
    try:
        import mlflow
        import mlflow.pytorch

        fm = payload.get("fused_model", {})
        mlflow.log_metrics(
            {
                "accuracy_mean": fm.get("accuracy_mean", 0),
                "accuracy_std": fm.get("accuracy_std", 0),
                "macro_f1_mean": fm.get("macro_f1_mean", 0),
                "macro_f1_std": fm.get("macro_f1_std", 0),
                "weighted_f1_mean": fm.get("weighted_f1_mean", 0),
                "weighted_f1_std": fm.get("weighted_f1_std", 0),
            }
        )

        metrics_path = base_dir / "data" / "processed" / "results" / "metrics.json"
        if metrics_path.exists():
            mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

        best_ckpt = base_dir / "data" / "processed" / "checkpoints" / "best_model.pt"
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoints")

        # Log PyTorch Model artifact
        if best_model is not None:
            try:
                mlflow.pytorch.log_model(
                    pytorch_model=best_model,
                    artifact_path="model",
                )
                print("[mlflow] Logged PyTorch model artifact to 'model/'")
            except Exception as m_err:
                print(f"[mlflow] PyTorch model logging warning: {m_err}")

        uri = mlflow.get_tracking_uri()
        info = {
            "run_id": run_id,
            "model_artifact_path": "model",
            "model_uri": f"runs:/{run_id}/model",
            "tracking_uri": uri,
            "experiment_name": os.getenv("MLFLOW_EXPERIMENT", "bionexus-cancer-subtype"),
            "metrics": {
                "accuracy_mean": fm.get("accuracy_mean", 0),
                "macro_f1_mean": fm.get("macro_f1_mean", 0),
            },
        }
        info_path = base_dir / "reports" / "experiment_info.json"
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info, indent=2))
        print(f"[mlflow] Saved experiment info to {info_path}")

        card_path = base_dir / "data" / "serving" / "model_card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text())
            card["mlflow_run_id"] = run_id
            card_path.write_text(json.dumps(card, indent=2))

        mlflow.end_run()
        print(f"[mlflow] Completed run {run_id}")
    except Exception as exc:
        print(f"[mlflow] finish failed: {exc}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(base_dir: Path):
    graph_dir = base_dir / "data" / "processed" / "graphs"
    final_dir = base_dir / "data" / "processed" / "final"
    serving_dir = base_dir / "data" / "serving"

    if (graph_dir / "rna_graph.pt").exists() and (graph_dir / "graph_node_order.csv").exists() and (final_dir / "labels_final.csv").exists():
        rna_graph = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
        mirna_graph = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
        meth_graph = torch.load(graph_dir / "meth_graph.pt", weights_only=False)
        node_order = pd.read_csv(graph_dir / "graph_node_order.csv")["patient_id"].tolist()
        labels_df = pd.read_csv(final_dir / "labels_final.csv").set_index("patient_id")
        labels_df = labels_df.loc[node_order]
        label_to_idx = {name: i for i, name in enumerate(PAM50_SUBTYPES)}
        y = labels_df["subtype"].map(label_to_idx).to_numpy()
        return rna_graph, mirna_graph, meth_graph, y.astype(int), node_order

    # Fallback / build from data/serving/
    print("[data] Constructing PyG multi-omics graphs from data/serving/...")
    from torch_geometric.data import Data

    patients_df = pd.read_csv(serving_dir / "patients.csv")
    node_order = patients_df["patient_id"].tolist()
    label_to_idx = {name: i for i, name in enumerate(PAM50_SUBTYPES)}
    y = patients_df["true_pam50"].map(label_to_idx).to_numpy()

    def _build_pyg_data(matrix_file: str, edges_file: str) -> Data:
        mat_df = pd.read_parquet(serving_dir / matrix_file).loc[node_order]
        edges_df = pd.read_parquet(serving_dir / edges_file)
        x_tensor = torch.tensor(mat_df.values, dtype=torch.float32)

        src = edges_df["src_idx"].values
        dst = edges_df["dst_idx"].values
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        w_col = "weight" if "weight" in edges_df.columns else "cosine_similarity"
        weights = edges_df[w_col].values if w_col in edges_df.columns else np.ones(len(src), dtype=np.float32)
        edge_attr = torch.tensor(weights, dtype=torch.float32)

        return Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)

    rna_graph = _build_pyg_data("rna_matrix.parquet", "rna_graph_edges.parquet")
    mirna_graph = _build_pyg_data("mirna_matrix.parquet", "mirna_graph_edges.parquet")
    meth_graph = _build_pyg_data("meth_matrix.parquet", "meth_graph_edges.parquet")

    # Cache to data/processed/graphs
    try:
        graph_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(rna_graph, graph_dir / "rna_graph.pt")
        torch.save(mirna_graph, graph_dir / "mirna_graph.pt")
        torch.save(meth_graph, graph_dir / "meth_graph.pt")
        pd.DataFrame({"patient_id": node_order}).to_csv(graph_dir / "graph_node_order.csv", index=False)
        pd.DataFrame({"patient_id": node_order, "subtype": patients_df["true_pam50"]}).to_csv(final_dir / "labels_final.csv", index=False)
        print(f"[data] Cached PyG graphs to {graph_dir}")
    except Exception as save_err:
        print(f"[data] Warning caching graphs: {save_err}")

    return rna_graph, mirna_graph, meth_graph, y.astype(int), node_order


def train_one_fold(
    rna_graph,
    mirna_graph,
    meth_graph,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    fold_num: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BioNexusClassifier(
        rna_in_features=rna_graph.x.shape[1],
        mirna_in_features=mirna_graph.x.shape[1],
        meth_in_features=meth_graph.x.shape[1],
    ).to(device)

    rna_graph, mirna_graph, meth_graph = (
        rna_graph.to(device),
        mirna_graph.to(device),
        meth_graph.to(device),
    )
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    # Class weights computed from TRAINING fold labels only
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(PAM50_SUBTYPES)), y=y[train_idx]
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        logits, _, _, _ = model(rna_graph, mirna_graph, meth_graph)
        loss = loss_fn(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(rna_graph, mirna_graph, meth_graph)
            val_preds = torch.argmax(logits[val_mask], dim=-1).cpu().numpy()
            val_true = y[val_idx]
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 20 == 0 or epochs_without_improvement == 0:
            tr_loss = loss.item()
            print(
                f"  [fold {fold_num}] epoch {epoch:3d} | loss={tr_loss:.4f} | val_f1={val_f1:.4f}"
            )

        if epochs_without_improvement >= cfg.early_stop_patience:
            print(
                f"  [fold {fold_num}] early stopping epoch {epoch} (best val_f1={best_val_f1:.4f})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _, _ = model(rna_graph, mirna_graph, meth_graph)
        val_preds = torch.argmax(logits[val_mask], dim=-1).cpu().numpy()
    val_true = y[val_idx]

    metrics = {
        "accuracy": accuracy_score(val_true, val_preds),
        "macro_f1": f1_score(val_true, val_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(val_true, val_preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(
            val_true, val_preds, labels=list(range(len(PAM50_SUBTYPES)))
        ),
    }
    return model, metrics


def main():
    BASE_DIR = Path(__file__).resolve().parents[2]  # repo root, from src/bio_nexus/
    cfg = TrainConfig()
    set_seed(cfg.seed)

    print("Loading data...")
    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(BASE_DIR)
    print(f"Loaded {len(y)} patients. Subtype distribution:")
    for name, idx in zip(PAM50_SUBTYPES, range(len(PAM50_SUBTYPES)), strict=False):
        print(f"  {name}: {(y == idx).sum()}")

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    fold_metrics = []
    best_fold_model = None
    best_fold_f1 = -1.0
    ckpt_dir = BASE_DIR / "data" / "processed" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extra_params = {
        "rna_in_features": int(rna_graph.x.shape[1]),
        "mirna_in_features": int(mirna_graph.x.shape[1]),
        "meth_in_features": int(meth_graph.x.shape[1]),
        "num_patients": int(len(y)),
    }
    mlflow_run_id = _maybe_start_mlflow(cfg, extra_params=extra_params, base_dir=BASE_DIR)

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print(f"\n=== Fold {fold_num}/{cfg.n_folds} ===")
        model, metrics = train_one_fold(
            rna_graph, mirna_graph, meth_graph, y, train_idx, val_idx, cfg, fold_num
        )
        print(
            f"  Fold {fold_num} result: accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, weighted_f1={metrics['weighted_f1']:.4f}"
        )
        fold_metrics.append(metrics)
        torch.save(model.state_dict(), ckpt_dir / f"fold_{fold_num}.pt")
        print(f"  Saved {ckpt_dir / f'fold_{fold_num}.pt'}")

        if mlflow_run_id:
            try:
                import mlflow
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

    print(f"\n=== Cross-Validation Summary ({cfg.n_folds} folds) ===")
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    wf1s = [m["weighted_f1"] for m in fold_metrics]
    print(f"Accuracy:     {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Macro F1:     {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"Weighted F1:  {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}")

    print(
        f"\nConfusion matrix (best fold, macro_f1={best_fold_f1:.4f}), "
        f"rows=true, cols=predicted, order={PAM50_SUBTYPES}:"
    )
    print(fold_metrics[np.argmax(f1s)]["confusion_matrix"])

    torch.save(best_fold_model.state_dict(), ckpt_dir / "best_model.pt")
    print(f"\nSaved best-fold model checkpoint to {ckpt_dir / 'best_model.pt'}")

    results_dir = BASE_DIR / "data" / "processed" / "results"
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
    print(f"Wrote {results_dir / 'metrics.json'}")
    _maybe_finish_mlflow(mlflow_run_id, payload, BASE_DIR, best_model=best_fold_model)


if __name__ == "__main__":
    main()
