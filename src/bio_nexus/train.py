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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(base_dir: Path):
    graph_dir = base_dir / "data" / "processed" / "graphs"
    final_dir = base_dir / "data" / "processed" / "final"

    rna_graph = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
    mirna_graph = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
    meth_graph = torch.load(graph_dir / "meth_graph.pt", weights_only=False)

    node_order = pd.read_csv(graph_dir / "graph_node_order.csv")["patient_id"].tolist()
    labels_df = pd.read_csv(final_dir / "labels_final.csv").set_index("patient_id")
    labels_df = labels_df.loc[node_order]  # align label order to graph node order EXACTLY

    label_to_idx = {name: i for i, name in enumerate(PAM50_SUBTYPES)}
    y = labels_df["subtype"].map(label_to_idx).to_numpy()

    if np.isnan(y).any():
        raise ValueError(
            "Some patients have a subtype not in PAM50_SUBTYPES - check labels_final.csv"
        )

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

    ckpt_dir = BASE_DIR / "data" / "processed" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_fold_model.state_dict(), ckpt_dir / "best_model.pt")
    print(f"\nSaved best-fold model checkpoint to {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
