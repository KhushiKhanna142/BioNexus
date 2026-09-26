"""
bio_nexus.interpretability.statistical_validation
---------------------------------------------------------
Rigorous statistical validation of gene attribution, replacing the
unstable "mean ratio" metric with a proper paired significance test.

For each of N patients (default 50), measures the ABSOLUTE logit change
(impact magnitude) caused by:
  (a) ablating the top-20 attribution-flagged genes
  (b) ablating 20 random genes (averaged over 20 trials)

Then runs a Wilcoxon signed-rank test (paired, non-parametric - suited
to small/skewed samples, robust to the noisy/reversed cases seen in a
raw ratio metric) testing whether (a) is systematically larger than (b)
across the patient population. Reports a p-value: the standard way to
state "this isn't due to chance" for a research paper or defense.

Usage:
    python -m bio_nexus.interpretability.statistical_validation
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from bio_nexus.models.classifier import BioNexusClassifier
from bio_nexus.train import load_data, set_seed
from bio_nexus.interpretability.attribution import integrated_gradients, top_attributed_features
from bio_nexus.interpretability.causal_ablation import get_target_logit, ablate_features


def measure_impact(model, graphs: list, target_layer: str, patient_idx: int,
                     top_k: int = 20, n_random_trials: int = 20, seed: int = 42):
    """Returns (top_k_impact, random_impact_mean) - both ABSOLUTE logit
    changes, so direction (increase vs decrease) doesn't obscure the
    comparison. Attribution's top-20 genes mix "pushes toward" and
    "pushes away" genes by design, so signed drops can go either way;
    magnitude of change is the cleaner, standard measure of impact."""
    set_seed(seed + patient_idx)  # vary random genes per patient, but reproducibly
    num_features = graphs[{"rna": 0, "mirna": 1, "meth": 2}[target_layer]].x.shape[1]

    attribution, pred_class, _ = integrated_gradients(model, graphs, target_layer, patient_idx, steps=50)
    top_features = top_attributed_features(attribution, list(range(num_features)), top_k=top_k)
    top_indices = [f[2] for f in top_features]

    orig_logit = get_target_logit(model, graphs, patient_idx, pred_class)

    top_ablated_logit = ablate_features(model, graphs, target_layer, patient_idx, top_indices, pred_class)
    top_impact = abs(orig_logit - top_ablated_logit)

    random_impacts = []
    for _ in range(n_random_trials):
        random_indices = list(np.random.choice(num_features, size=top_k, replace=False))
        random_logit = ablate_features(model, graphs, target_layer, patient_idx, random_indices, pred_class)
        random_impacts.append(abs(orig_logit - random_logit))

    return top_impact, float(np.mean(random_impacts))


def main():
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

    n_patients = min(50, len(node_order))
    print(f"Measuring attribution impact (logit-magnitude) across {n_patients} patients "
          f"(this involves {n_patients} x 21 forward passes - will take a while)...\n", flush=True)

    top_impacts, random_impacts = [], []
    for patient_idx in range(n_patients):
        top_impact, random_impact = measure_impact(model, graphs, target_layer="rna", patient_idx=patient_idx)
        top_impacts.append(top_impact)
        random_impacts.append(random_impact)
        if (patient_idx + 1) % 10 == 0:
            print(f"  processed {patient_idx + 1}/{n_patients} patients...", flush=True)

    top_impacts = np.array(top_impacts)
    random_impacts = np.array(random_impacts)

    n_top_greater = int((top_impacts > random_impacts).sum())

    stat, p_value = wilcoxon(top_impacts, random_impacts, alternative="greater")

    print(f"\n=== Statistical Validation Results ({n_patients} patients) ===")
    print(f"Median top-K attribution impact:    {np.median(top_impacts):.4f}")
    print(f"Median random-gene impact:          {np.median(random_impacts):.4f}")
    print(f"Mean top-K attribution impact:      {np.mean(top_impacts):.4f}")
    print(f"Mean random-gene impact:            {np.mean(random_impacts):.4f}")
    print(f"Patients where top-K > random:      {n_top_greater}/{n_patients} "
          f"({n_top_greater/n_patients:.1%})")
    print(f"\nWilcoxon signed-rank test (one-sided, top-K > random): "
          f"statistic={stat:.2f}, p-value={p_value:.6f}")

    if p_value < 0.001:
        print("Result: HIGHLY SIGNIFICANT (p < 0.001) - attribution-flagged genes cause "
              "significantly larger prediction changes than random genes. This is strong "
              "evidence gradient attribution is finding causally relevant genes, not noise.")
    elif p_value < 0.05:
        print("Result: SIGNIFICANT (p < 0.05) - attribution-flagged genes cause significantly "
              "larger prediction changes than random genes.")
    else:
        print("Result: NOT statistically significant at p < 0.05 - the evidence that attribution "
              "outperforms random gene selection is not strong enough at this sample size. "
              "Consider a larger patient sample or reviewing the attribution method.")

    out_dir = BASE_DIR / "data" / "processed" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "patient_id": node_order[:n_patients],
        "top_k_impact": top_impacts,
        "random_impact_mean": random_impacts,
    }).to_csv(out_dir / "attribution_statistical_validation.csv", index=False)
    print(f"\nSaved per-patient results: {out_dir / 'attribution_statistical_validation.csv'}")


if __name__ == "__main__":
    main()