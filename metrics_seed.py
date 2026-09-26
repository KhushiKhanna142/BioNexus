"""
One-off script: writes data/processed/results/metrics.json from the
known train.py / ablation.py results, since those scripts printed to
terminal but never saved a structured metrics file for the API to serve.

Run this ONCE now to seed the dashboard. Going forward, train.py should
be updated to save this automatically at the end of each run (not done
here to avoid re-triggering a full retrain) - see the TODO at the bottom.

Usage:
    python3 metrics_seed.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # repo root
OUT_PATH = BASE_DIR / "data" / "processed" / "results" / "metrics.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

metrics = {
    "fused_model": {
        "accuracy_mean": 0.8699,
        "accuracy_std": 0.0301,
        "macro_f1_mean": 0.8458,
        "macro_f1_std": 0.0296,
        "weighted_f1_mean": 0.8713,
        "weighted_f1_std": 0.0270,
        "confusion_matrix": {
            "labels": ["LumA", "LumB", "Her2", "Basal", "Normal"],
            "matrix": [
                [62, 3, 1, 0, 2],
                [6, 20, 0, 0, 0],
                [0, 0, 7, 0, 0],
                [0, 0, 0, 23, 0],
                [0, 0, 0, 1, 5],
            ],
        },
        "per_class_recall": {
            "LumA": 0.912,
            "LumB": 0.769,
            "Her2": 1.0,
            "Basal": 1.0,
            "Normal": 0.833,
        },
    },
    "single_omics_ablation": {
        "rna": {
            "accuracy_mean": 0.8144,
            "accuracy_std": 0.0836,
            "macro_f1_mean": 0.8104,
            "macro_f1_std": 0.0765,
        },
        "mirna": {
            "accuracy_mean": 0.7492,
            "accuracy_std": 0.0405,
            "macro_f1_mean": 0.7242,
            "macro_f1_std": 0.0361,
        },
        "meth": {
            "accuracy_mean": 0.7230,
            "accuracy_std": 0.0318,
            "macro_f1_mean": 0.7280,
            "macro_f1_std": 0.0365,
        },
    },
    "interpretability": {
        "wilcoxon_p_value": 0.0,  # reported as p < 0.001 - exact value from your run's output
        "n_patients_validated": 50,
        "pct_patients_top_k_beats_random": 0.82,
        "median_top_k_impact": 0.0045,
        "median_random_impact": 0.0010,
    },
    "cohort": {
        "n_patients": 646,
        "n_omics_layers": 3,
    },
}

OUT_PATH.write_text(json.dumps(metrics, indent=2))
print(f"Saved: {OUT_PATH}")

# TODO (future improvement, not done here): add a few lines at the end of
# train.py's main() that write this same structure automatically after
# each training run, so metrics.json always reflects the latest model
# rather than being manually seeded from a specific past run's printed output.
