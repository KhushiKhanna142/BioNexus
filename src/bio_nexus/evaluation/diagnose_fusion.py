"""
bio_nexus.evaluation.diagnose_fusion
------------------------------------------
Loads the trained best-fold checkpoint and inspects what the fusion
module actually learned: are the per-modality pooling weights close to
uniform (33/33/33 - meaning fusion ISN'T learning to trust one layer
over another), or has it learned to downweight the weaker layers (RNA)?

Usage:
    python -m bio_nexus.evaluation.diagnose_fusion
"""

from pathlib import Path

import torch

from bio_nexus.models.classifier import BioNexusClassifier
from bio_nexus.train import load_data


def main():
    BASE_DIR = Path(__file__).resolve().parents[3]
    ckpt_path = BASE_DIR / "data" / "processed" / "checkpoints" / "best_model.pt"

    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(BASE_DIR)

    model = BioNexusClassifier(
        rna_in_features=rna_graph.x.shape[1],
        mirna_in_features=mirna_graph.x.shape[1],
        meth_in_features=meth_graph.x.shape[1],
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        logits, fused, attn_weights, pool_weights = model(rna_graph, mirna_graph, meth_graph)

    pool_weights = pool_weights.numpy()  # [646, 3] -> RNA, miRNA, methylation

    print("=== Fusion pooling weights across all 646 patients ===")
    print(
        "(if fusion learned to trust miRNA/meth more than RNA, "
        "RNA's mean should be lower)\n"
    )
    layer_names = ["rna", "mirna", "meth"]
    for i, name in enumerate(layer_names):
        col = pool_weights[:, i]
        print(
            f"{name:8s}: mean={col.mean():.4f}  std={col.std():.4f}  "
            f"min={col.min():.4f}  max={col.max():.4f}"
        )

    uniform = 1 / 3
    print(f"\n(Uniform/no-preference baseline would be {uniform:.4f} for each layer)")

    max_gap = pool_weights.mean(axis=0).max() - pool_weights.mean(axis=0).min()
    print(f"\nLargest gap between mean weights: {max_gap:.4f}")
    if max_gap < 0.05:
        print(
            "[FINDING] Weights are close to uniform - fusion has NOT learned strong "
            "per-modality preference. This suggests the fusion mechanism isn't "
            "effectively discounting the weaker RNA layer."
        )
    else:
        print("[FINDING] Fusion HAS learned differentiated weights across modalities.")

    # Correlate: does high RNA-weight correlate with WORSE predictions?
    preds = torch.argmax(logits, dim=-1).numpy()
    correct = preds == y
    rna_weight = pool_weights[:, 0]

    print(f"\nMean RNA pooling weight for CORRECT predictions:   {rna_weight[correct].mean():.4f}")
    print(f"Mean RNA pooling weight for INCORRECT predictions: {rna_weight[~correct].mean():.4f}")
    print(
        "(if RNA weight is similar in both groups, the model isn't successfully "
        "using RNA-weight as a signal of unreliability)"
    )


if __name__ == "__main__":
    main()
