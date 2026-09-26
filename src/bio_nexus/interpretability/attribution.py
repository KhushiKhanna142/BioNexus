"""
bio_nexus.interpretability.attribution
--------------------------------------------
Gradient-based feature attribution (Integrated Gradients) for the trained
BioNexusClassifier. Answers: "which genes/probes/miRNAs drove THIS
patient's prediction?"

Implemented directly (not via captum) because the model takes 3 separate
graph inputs sharing fixed graph structure - IG here interpolates ONE
layer's raw features between a zero baseline and the real input, while
holding the other two layers' features AND all graph structure (edge_index,
edge_attr) fixed, then integrates the gradient of the target patient's
predicted-class logit along that path.

IMPORTANT: attribution captures how much each of patient P's OWN features
contributed to patient P's prediction. Because GraphGPS uses message-passing,
neighbor patients' features also influence P's prediction (that's the whole
point of the graph) - this module reports only the "own-feature" row of the
gradient, which is the conventional, most directly actionable definition of
per-patient feature importance.

Usage:
    python -m bio_nexus.interpretability.attribution
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.train import load_data

LAYER_TO_INDEX = {"rna": 0, "mirna": 1, "meth": 2}


def integrated_gradients(
    model,
    graphs: list,
    target_layer: str,
    patient_idx: int,
    target_class: int = None,
    steps: int = 50,
):
    """
    graphs: [rna_graph, mirna_graph, meth_graph] (order matters - matches model.forward)
    target_layer: "rna", "mirna", or "meth" - which layer's features to attribute
    target_class: class index to explain; defaults to the model's own predicted class
    Returns: (attribution [num_features] for patient_idx's own row, predicted_class, confidence)
    """
    model.eval()
    layer_idx = LAYER_TO_INDEX[target_layer]
    target_graph = graphs[layer_idx]
    x_real = target_graph.x.clone()
    baseline = torch.zeros_like(x_real)  # zero-expression / zero-signal baseline

    if target_class is None:
        with torch.no_grad():
            logits, _, _, _ = model(*graphs)
            target_class = torch.argmax(logits[patient_idx]).item()

    alphas = torch.linspace(0, 1, steps)
    accumulated_grad = torch.zeros_like(x_real)

    for alpha in alphas:
        x_interp = (baseline + alpha * (x_real - baseline)).clone().requires_grad_(True)

        interp_graphs = list(graphs)
        interp_target_graph = copy.copy(
            target_graph
        )  # shallow copy - reuse same edge_index/edge_attr
        interp_target_graph.x = x_interp
        interp_graphs[layer_idx] = interp_target_graph

        logits, _, _, _ = model(*interp_graphs)
        target_logit = logits[patient_idx, target_class]

        grad = torch.autograd.grad(target_logit, x_interp, retain_graph=False)[0]
        accumulated_grad += grad

    avg_grad = accumulated_grad / steps
    attribution_matrix = avg_grad * (x_real - baseline)  # [num_patients, num_features]
    patient_attribution = attribution_matrix[patient_idx]  # this patient's own-feature row

    with torch.no_grad():
        logits, _, _, _ = model(*graphs)
        confidence = torch.softmax(logits[patient_idx], dim=-1)[target_class].item()

    return patient_attribution.detach().numpy(), target_class, confidence


def top_attributed_features(attribution: np.ndarray, feature_names: list, top_k: int = 20):
    """Returns top_k (feature_name, attribution_score) pairs, sorted by |attribution| descending."""
    idx_sorted = np.argsort(-np.abs(attribution))[:top_k]
    return [(feature_names[i], float(attribution[i]), int(i)) for i in idx_sorted]


def load_feature_names(base_dir: Path, layer_name: str) -> list:
    path = base_dir / "data" / "processed" / "final" / f"{layer_name}_matrix_final.csv"
    header = pd.read_csv(path, index_col=0, nrows=0)
    return list(header.columns)


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

    rna_names = load_feature_names(BASE_DIR, "rna")

    # Demo: explain the RNA-driving genes for the first 3 patients
    for patient_idx in range(3):
        attribution, pred_class, confidence = integrated_gradients(
            model, graphs, target_layer="rna", patient_idx=patient_idx, steps=50
        )
        top_genes = top_attributed_features(attribution, rna_names, top_k=10)

        print(
            f"\n=== Patient {node_order[patient_idx]} (true: {PAM50_SUBTYPES[y[patient_idx]]}, "
            f"predicted: {PAM50_SUBTYPES[pred_class]}, confidence: {confidence:.1%}) ==="
        )
        print("Top 10 RNA genes by attribution (gene, score, feature_index):")
        for name, score, _idx in top_genes:
            direction = "pushes toward" if score > 0 else "pushes away from"
            print(f"  {name:20s} {score:+.4f}  ({direction} predicted class)")
