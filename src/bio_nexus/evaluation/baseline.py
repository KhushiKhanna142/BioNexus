"""
bio_nexus.evaluation.baselines
------------------------------------
Fair benchmark of classical models on the SAME 646-patient cohort, using
the SAME stratified 5-fold splits as train.py (StratifiedKFold, 5 folds,
shuffle, seed 42, labels encoded LumA/LumB/Her2/Basal/Normal -> 0..4),
so the numbers are directly comparable to the BioNexus fused result.

What it does differently from a quick baseline run:
  1. Prints the value range of every omics layer first, so a scale problem
     is visible instead of guessed at.
  2. Standardizes features INSIDE each training fold (never on all data),
     for the models that are scale-sensitive (LogReg, SVM, MLP).
  3. Includes a majority-class baseline, so "accuracy" is always read
     against the ~53% you get from always predicting LumA.
  4. Runs each model on RNA only, miRNA only, methylation only, and all
     three layers concatenated.

Models are lightly configured, not tuned - good enough to see whether they
learn, not a tuning contest.

Usage:
    pip install xgboost lightgbm scikit-learn
    python -m bio_nexus.evaluation.baselines
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Same order as PAM50_SUBTYPES in models/classifier.py - keeps fold splits identical to train.py
SUBTYPES = ["LumA", "LumB", "Her2", "Basal", "Normal"]
SEED = 42
N_FOLDS = 5


def load_data(base_dir: Path):
    final_dir = base_dir / "data" / "processed" / "final"
    mats = {
        name: pd.read_csv(final_dir / f"{name}_matrix_final.csv", index_col=0)
        for name in ["rna", "mirna", "meth"]
    }
    labels = pd.read_csv(final_dir / "labels_final.csv").set_index("patient_id")

    ref_index = mats["rna"].index
    for name, m in mats.items():
        if not m.index.equals(ref_index):
            raise ValueError(f"{name} matrix patient order differs from rna - cannot align layers")

    labels = labels.loc[ref_index]
    y = labels["subtype"].map({s: i for i, s in enumerate(SUBTYPES)})
    if y.isna().any():
        raise ValueError("Some labels are not in the PAM50 list - check labels_final.csv")
    return mats, y.to_numpy().astype(int)


def describe_scales(mats: dict):
    print("=== Feature value ranges per layer (look for huge max values) ===")
    print(f"{'layer':<8}{'features':>9}{'min':>11}{'p99.9':>11}{'max':>13}{'NaN/inf':>9}{'zero-var':>10}")
    for name, m in mats.items():
        v = m.to_numpy(dtype=np.float64)
        bad = int((~np.isfinite(v)).sum())
        zero_var = int((np.nanstd(v, axis=0) == 0).sum())
        print(f"{name:<8}{v.shape[1]:>9}{np.nanmin(v):>11.3f}{np.nanpercentile(v, 99.9):>11.3f}"
              f"{np.nanmax(v):>13.3f}{bad:>9}{zero_var:>10}")
    print()


def make_models():
    return {
        "majority-class": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
        ),
        "svm-rbf": make_pipeline(
            StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced")
        ),
        "random-forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.3, tree_method="hist", n_jobs=-1, random_state=SEED,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.3, class_weight="balanced",
            random_state=SEED, verbose=-1,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-2, early_stopping=True,
                          max_iter=500, random_state=SEED),
        ),
    }


def fit_predict(name, model, X_tr, y_tr, X_te):
    if name == "xgboost":
        # XGBoost has no class_weight argument - pass balanced sample weights instead
        model.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    else:
        model.fit(X_tr, y_tr)
    return model.predict(X_te)


def main():
    base_dir = Path(__file__).resolve().parents[3]
    mats, y = load_data(base_dir)
    describe_scales(mats)

    counts = np.bincount(y, minlength=len(SUBTYPES))
    print("Class counts:", {s: int(c) for s, c in zip(SUBTYPES, counts)})
    print(f"Majority-class accuracy to beat: {counts.max() / counts.sum():.3f}\n")

    feature_sets = {
        "rna": mats["rna"].to_numpy(dtype=np.float32),
        "mirna": mats["mirna"].to_numpy(dtype=np.float32),
        "meth": mats["meth"].to_numpy(dtype=np.float32),
    }
    feature_sets["all3"] = np.hstack([feature_sets["rna"], feature_sets["mirna"], feature_sets["meth"]])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))

    rows = []
    for fs_name, X in feature_sets.items():
        for model_name in make_models():
            accs, f1s = [], []
            t0 = time.time()
            for train_idx, test_idx in folds:
                model = make_models()[model_name]  # fresh model each fold
                preds = fit_predict(model_name, model, X[train_idx], y[train_idx], X[test_idx])
                accs.append(accuracy_score(y[test_idx], preds))
                f1s.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
            rows.append({
                "feature_set": fs_name, "model": model_name,
                "acc_mean": np.mean(accs), "acc_std": np.std(accs),
                "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
            })
            print(f"[{fs_name:<5}] {model_name:<15} acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}  "
                  f"macro-F1={np.mean(f1s):.3f}  ({time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows)

    for metric, label in [("acc", "Accuracy"), ("f1", "Macro F1")]:
        table = results.assign(
            cell=results[f"{metric}_mean"].map("{:.3f}".format) + " +/- " + results[f"{metric}_std"].map("{:.3f}".format)
        ).pivot(index="model", columns="feature_set", values="cell")
        print(f"\n=== {label} (mean +/- std over {N_FOLDS} folds) ===")
        print(table[["rna", "mirna", "meth", "all3"]].to_string())

    out_dir = base_dir / "data" / "processed" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "baseline_results.csv", index=False)
    print(f"\nSaved: {out_dir / 'baseline_results.csv'}")
    print("Compare against the BioNexus fused result from train.py "
          "(same folds): accuracy 0.8699, macro-F1 0.8458.")


if __name__ == "__main__":
    main()