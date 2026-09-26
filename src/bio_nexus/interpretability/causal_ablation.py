"""
bio_nexus.interpretability.causal_ablation
--------------------------------------------------
Validates gradient-based attribution CAUSALLY, not just correlationally:
takes the top-K genes flagged by attribution.py for a patient, zeroes
them out (ablates them), and measures how much the model's confidence
in its original prediction drops. Compares this against ablating K
RANDOM genes (multiple trials) as a control.

If top-attributed genes cause a much bigger confidence drop than random
genes, that's direct causal evidence the attribution is finding genes
the model actually relies on - not just genes that happen to correlate
with the prediction. This directly addresses the "causal ablation
validation" gap identified in the project's literature review.

Usage:
    python -m bio_nexus.interpretability.causal_ablation
"""

import copy
from pathlib import Path

import numpy as np
import torch

from bio_nexus.interpretability.attribution import (
    LAYER_TO_INDEX,
    integrated_gradients,
    top_attributed_features,
)
from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.train import load_data, set_seed


def get_prediction_confidence(model, graphs: list, patient_idx: int, target_class: int) -> float:
    with torch.no_grad():
        logits, _, _, _ = model(*graphs)
        return torch.softmax(logits[patient_idx], dim=-1)[target_class].item()


def get_target_logit(model, graphs: list, patient_idx: int, target_class: int) -> float:
    """Raw pre-softmax logit for target_class - doesn't saturate near 0/1 the way
    softmax probability does, so it stays sensitive even for very confident
    predictions (e.g. 99% probability). Use this for ablation drop measurement;
    use softmax probability only for human-readable confidence reporting."""
    with torch.no_grad():
        logits, _, _, _ = model(*graphs)
        return logits[patient_idx, target_class].item()


def ablate_features(
    model,
    graphs: list,
    target_layer: str,
    patient_idx: int,
    feature_indices: list,
    target_class: int,
) -> float:
    """Zeroes out the given feature indices for ONE patient's row in the
    target layer, re-runs the model, and returns the new LOGIT (not
    softmax probability) for target_class. Logits are used instead of
    probability because softmax saturates near 0/1 for confident
    predictions, making probability-space drops near-zero and noisy
    even when the ablated features genuinely matter. Other patients'
    features and all graph structure are left untouched - this tests
    ONLY the effect of patient_idx's own ablated features (consistent
    with attribution.py's own-row convention)."""
    layer_idx = LAYER_TO_INDEX[target_layer]
    ablated_graphs = list(graphs)
    target_graph = copy.copy(graphs[layer_idx])

    x_ablated = target_graph.x.clone()
    x_ablated[patient_idx, feature_indices] = 0.0
    target_graph.x = x_ablated
    ablated_graphs[layer_idx] = target_graph

    return get_target_logit(model, ablated_graphs, patient_idx, target_class)


def causal_ablation_check(
    model,
    graphs: list,
    target_layer: str,
    patient_idx: int,
    top_k: int = 20,
    n_random_trials: int = 20,
    seed: int = 42,
):
    """Compares LOGIT drop from ablating top-K attributed genes vs
    K random genes (averaged over n_random_trials). Logit-space avoids
    the softmax-saturation problem where confident predictions (>95%)
    show near-zero probability movement regardless of whether the
    ablated features actually matter. Returns a dict summary."""
    set_seed(seed)
    num_features = graphs[LAYER_TO_INDEX[target_layer]].x.shape[1]

    attribution, pred_class, orig_confidence = integrated_gradients(
        model, graphs, target_layer, patient_idx, steps=50
    )
    top_features = top_attributed_features(attribution, list(range(num_features)), top_k=top_k)
    top_indices = [f[2] for f in top_features]

    orig_logit = get_target_logit(model, graphs, patient_idx, pred_class)

    top_ablated_logit = ablate_features(
        model, graphs, target_layer, patient_idx, top_indices, pred_class
    )
    top_drop = orig_logit - top_ablated_logit

    random_drops = []
    for _ in range(n_random_trials):
        random_indices = list(np.random.choice(num_features, size=top_k, replace=False))
        random_logit = ablate_features(
            model, graphs, target_layer, patient_idx, random_indices, pred_class
        )
        random_drops.append(orig_logit - random_logit)

    return {
        "patient_idx": patient_idx,
        "predicted_class": PAM50_SUBTYPES[pred_class],
        "original_confidence": orig_confidence,
        "original_logit": orig_logit,
        "top_k_logit_drop": top_drop,
        "random_logit_drop_mean": float(np.mean(random_drops)),
        "random_logit_drop_std": float(np.std(random_drops)),
        "ratio_top_vs_random": top_drop / (abs(np.mean(random_drops)) + 1e-4),
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[3]
    ckpt_path = BASE_DIR / "data" / "processed" / "checkpoints" / "best_model.pt"

    rna_graph, mirna_graph, meth_graph, y, node_order = load_data(BASE_DIR)
    graphs = [rna_graph, mirna_graph, meth_graph]

    model = BioNexusClassifier(
        rna_in_features=rna_graph.x.shape[1],
        mirna_in_features=mirna_graph.x.shape[1],
        meth_in_features=meth_graph.x.shape[1],
    )
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    print("Causal ablation validation: top-20 attributed RNA genes vs 20 random genes")
    print(
        "(measured in LOGIT space, not softmax probability, to avoid saturation)\n"
    )
    print(
        f"{'Patient':<15} {'Predicted':<10} {'Confidence':<12} {'Top-K Logit Drop':<18} "
        f"{'Random Logit Drop (mean+/-std)':<32} {'Ratio':<8}"
    )

    all_ratios = []
    for patient_idx in range(10):  # demo on first 10 patients
        result = causal_ablation_check(
            model, graphs, target_layer="rna", patient_idx=patient_idx, top_k=20, n_random_trials=20
        )
        all_ratios.append(result["ratio_top_vs_random"])
        p_id = node_order[patient_idx]
        pred_cls = result["predicted_class"]
        conf = f"{result['original_confidence']:.1%}"
        top_drop = f"{result['top_k_logit_drop']:+.4f}"
        m_val, s_val = result['random_logit_drop_mean'], result['random_logit_drop_std']
        rand_drop = f"{m_val:+.4f} +/- {s_val:.4f}"
        ratio_str = f"{result['ratio_top_vs_random']:.2f}x"
        print(f"{p_id:<15} {pred_cls:<10} {conf:<12} {top_drop:<18} {rand_drop:<32} {ratio_str:<8}")

    mean_ratio = np.mean(all_ratios)
    print(f"\nMean ratio (top-K / random logit drop) over 10 patients: {mean_ratio:.2f}x")
    print(
        "(Ratio > 1 means attribution-flagged genes cause a bigger LOGIT drop than random genes "
        "when ablated. Logit space avoids the softmax-saturation issue where confident predictions "
        "showed near-zero, noisy probability movement regardless of which genes were removed.)"
    )
