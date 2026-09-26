"""
bio_nexus.evaluation.ablation
-----------------------------------
Single-omics ablation study: trains the SAME GraphGPS encoder + classifier
architecture using only ONE omics layer at a time (RNA-only, miRNA-only,
methylation-only - no fusion), using the IDENTICAL stratified 5-fold CV
splits as the full fused model, for a direct apples-to-apples comparison.

This empirically tests whether multi-omics fusion actually helps THIS
architecture on THIS cohort - directly addressing the "causal ablation
validation" gap identified in the literature review, rather than just
citing that other papers found multi-omics beneficial.

Usage:
    python -m bio_nexus.evaluation.ablation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from bio_nexus.encoding.graph_transformer import GraphGPSEncoder, GraphTransformerConfig
from bio_nexus.models.classifier import PAM50_SUBTYPES
from bio_nexus.train import TrainConfig, load_data, set_seed


class SingleOmicsClassifier(nn.Module):
    """Same encoder architecture as the full model, but for ONE omics
    layer only - no fusion module, since there's nothing to fuse.
    Encoder embedding -> classifier head directly."""

    def __init__(
        self,
        in_features: int,
        encoder_cfg: GraphTransformerConfig,
        num_classes: int = 5,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = GraphGPSEncoder(in_features, encoder_cfg)
        self.classifier_head = nn.Sequential(
            nn.Linear(encoder_cfg.embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, graph):
        emb = self.encoder(graph.x, graph.edge_index, graph.edge_attr)
        logits = self.classifier_head(emb)
        return logits


def train_one_fold_single(
    graph,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg: TrainConfig,
    layer_name: str,
    fold_num: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SingleOmicsClassifier(graph.x.shape[1], GraphTransformerConfig()).to(device)
    graph = graph.to(device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    train_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(len(y), dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(len(PAM50_SUBTYPES)), y=y[train_idx]
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_f1, best_state, epochs_without_improvement = -1.0, None, 0

    for _epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(graph)
        loss = loss_fn(logits[train_mask], y_t[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph)
            val_preds = torch.argmax(logits[val_mask], dim=-1).cpu().numpy()
            val_f1 = f1_score(y[val_idx], val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1, best_state, epochs_without_improvement = (
                val_f1,
                {k: v.clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = torch.argmax(model(graph)[val_mask], dim=-1).cpu().numpy()

    return {
        "accuracy": accuracy_score(y[val_idx], val_preds),
        "macro_f1": f1_score(y[val_idx], val_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y[val_idx], val_preds, average="weighted", zero_division=0),
    }


def run_ablation_for_layer(graph, y, cfg, layer_name):
    print(f"\n--- {layer_name.upper()}-only ---")
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_metrics = []
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        m = train_one_fold_single(graph, y, train_idx, val_idx, cfg, layer_name, fold_num)
        print(f"  fold {fold_num}: accuracy={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
        fold_metrics.append(m)
    return fold_metrics


def main():
    BASE_DIR = Path(__file__).resolve().parents[3]
    cfg = TrainConfig()
    set_seed(cfg.seed)

    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(BASE_DIR)
    print(
        f"Running single-omics ablation on {len(y)} patients, {cfg.n_folds}-fold CV "
        f"(SAME folds/seed as the fused model, for fair comparison)"
    )

    results = {}
    for layer_name, graph in [("rna", rna_graph), ("mirna", mirna_graph), ("meth", meth_graph)]:
        set_seed(cfg.seed)  # reset seed before each layer so fold splits match exactly
        results[layer_name] = run_ablation_for_layer(graph, y, cfg, layer_name)

    print("\n\n=== Ablation Summary: Single-Omics vs Fused ===")
    print(f"{'Layer':<12} {'Accuracy':<20} {'Macro F1':<20} {'Weighted F1':<20}")

    summary_rows = []
    for layer_name, fold_metrics in results.items():
        accs = [m["accuracy"] for m in fold_metrics]
        f1s = [m["macro_f1"] for m in fold_metrics]
        wf1s = [m["weighted_f1"] for m in fold_metrics]
        print(
            f"{layer_name:<12} {np.mean(accs):.4f} +/- {np.std(accs):.4f}   "
            f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}   {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}"
        )
        summary_rows.append(
            {
                "layer": layer_name,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "macro_f1_mean": np.mean(f1s),
                "macro_f1_std": np.std(f1s),
                "weighted_f1_mean": np.mean(wf1s),
                "weighted_f1_std": np.std(wf1s),
            }
        )

    print(f"{'fused (all 3)':<12} see train.py output - run separately for the full-fusion number")
    print(
        "\nNOTE: compare each row above against your train.py fused-model result "
        "(Accuracy 0.7387, Macro F1 0.7114) to see whether fusion actually helps "
        "over each single omics layer alone."
    )

    out_dir = BASE_DIR / "data" / "processed" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_dir / "ablation_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
