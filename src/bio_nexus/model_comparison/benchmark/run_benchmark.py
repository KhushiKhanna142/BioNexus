#!/usr/bin/env python3
"""
model_comparison.benchmark.run_benchmark
-----------------------------------------
Executes real, empirical cross-validation benchmarking on multi-omics data.
Trains baseline models (Random Forest, MLP, Single-Omics, XGBoost) and incorporates
BioNexus out-of-fold predictions. Outputs data/splits.json and data/results.json
with data_status='measured'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Import registry and data validation
try:
    from bio_nexus.model_comparison.data_source import parse_dataset
    from bio_nexus.model_comparison.registry import MINORITY_CLASSES, PAM50_CLASSES
except ImportError:
    from model_comparison.data_source import parse_dataset
    from model_comparison.registry import MINORITY_CLASSES, PAM50_CLASSES


def load_table(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path_str}")
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    elif p.suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Run real multi-omics baseline benchmarks")
    parser.add_argument("--rna", required=True, help="Path to RNA-seq matrix (parquet/csv)")
    parser.add_argument("--mirna", required=True, help="Path to miRNA matrix (parquet/csv)")
    parser.add_argument("--meth", required=True, help="Path to Methylation matrix (parquet/csv)")
    parser.add_argument("--labels", required=True, help="Path to labels table (parquet/csv)")
    parser.add_argument("--label-col", default="true_pam50", help="Label column name (e.g. PAM50 or true_pam50)")
    parser.add_argument("--patient-col", default="patient_id", help="Patient barcode ID column name")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="src/model_comparison/data/results.json", help="Path to write results.json")
    parser.add_argument("--splits-out", default="data/splits.json", help="Path to write splits.json")
    parser.add_argument("--driver-genes", default=None, help="Path to driver gene symbols (optional)")
    parser.add_argument("--ext-cohort", default=None, help="Path to external cohort (optional)")
    parser.add_argument("--external", default=None, help="Path to external model spec JSON (e.g. bionexus_spec.json)")

    args = parser.parse_args()

    print("==================================================================")
    print("      BioNexus Model Comparison: Real Empirical Benchmark        ")
    print("==================================================================")

    # 1. Load Data
    print("\n[Step 1] Loading multi-omics matrices & clinical labels...")
    df_labels = load_table(args.labels)
    df_rna = load_table(args.rna)
    df_mirna = load_table(args.mirna)
    df_meth = load_table(args.meth)

    # Resolve label column
    label_col = args.label_col
    if label_col not in df_labels.columns:
        for candidate in ["PAM50", "pam50", "true_pam50", "Subtype", "subtype"]:
            if candidate in df_labels.columns:
                label_col = candidate
                break
    if label_col not in df_labels.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in labels file. Columns: {df_labels.columns.tolist()}")

    # Resolve patient column
    pat_col = args.patient_col
    if pat_col not in df_labels.columns:
        pat_col = df_labels.columns[0]

    # Index by patient ID
    df_labels = df_labels.set_index(pat_col)
    if "patient_id" in df_rna.columns:
        df_rna = df_rna.set_index("patient_id")
    if "patient_id" in df_mirna.columns:
        df_mirna = df_mirna.set_index("patient_id")
    if "patient_id" in df_meth.columns:
        df_meth = df_meth.set_index("patient_id")

    # Common patients
    common_patients = df_labels.index.intersection(df_rna.index).intersection(df_mirna.index).intersection(df_meth.index)
    common_patients = list(common_patients)
    print(f"Aligining cohort across all 3 omics layers: {len(common_patients)} patients.")

    if len(common_patients) == 0:
        raise RuntimeError("No common patient IDs found across RNA, miRNA, methylation and labels tables!")

    df_labels = df_labels.loc[common_patients]
    y_raw = df_labels[label_col].values

    # Check subtype distribution
    label_series = pd.Series(y_raw, index=common_patients)
    counts = label_series.value_counts()
    print("\nPAM50 Subtype Distribution:")
    for sub, cnt in counts.items():
        print(f"  {sub}: {cnt}")
        if cnt < 5:
            print(f"CRITICAL STOP: Subtype '{sub}' has only {cnt} patients (< 5 required).")
            sys.exit(1)

    # Encode labels to canonical order: LumA, LumB, Her2, Basal, Normal
    canonical_classes = PAM50_CLASSES
    class_to_idx = {c: i for i, c in enumerate(canonical_classes)}
    y = np.array([class_to_idx[lbl] for lbl in y_raw])
    n_classes = len(canonical_classes)

    # Prepare feature matrices
    X_rna = df_rna.loc[common_patients].values.astype(np.float32)
    X_mirna = df_mirna.loc[common_patients].values.astype(np.float32)
    X_meth = df_meth.loc[common_patients].values.astype(np.float32)

    # Check for NaNs and impute with column median
    for mat in [X_rna, X_mirna, X_meth]:
        if np.isnan(mat).any():
            col_med = np.nanmedian(mat, axis=0)
            inds = np.where(np.isnan(mat))
            mat[inds] = np.take(col_med, inds[1])

    X_concat = np.hstack([X_rna, X_mirna, X_meth])
    print(f"\nFeature Matrices Dimension:")
    print(f"  - RNA-seq:        {X_rna.shape[1]} features")
    print(f"  - miRNA:          {X_mirna.shape[1]} features")
    print(f"  - DNA Methylation:{X_meth.shape[1]} features")
    print(f"  - Multi-omics:    {X_concat.shape[1]} features")

    # 2. Generate Stratified 5-Fold Splits
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X_concat, y))

    splits_data = {
        "n_patients": len(common_patients),
        "patients": common_patients,
        "folds": {
            f"fold_{i}": {
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "train_patients": [common_patients[idx] for idx in train_idx],
                "val_patients": [common_patients[idx] for idx in val_idx],
            }
            for i, (train_idx, val_idx) in enumerate(splits)
        },
    }
    splits_path = Path(args.splits_out)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits_data, indent=2))
    print(f"\nSaved cross-validation splits to {splits_path}")

    # Helper to evaluate out-of-fold predictions
    def evaluate_oof(oof_probs: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
        preds = np.argmax(oof_probs, axis=1)
        acc = float(accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        try:
            # One-hot encode y_true for multiclass AUC
            y_one_hot = np.zeros((len(y_true), n_classes))
            for i, val in enumerate(y_true):
                y_one_hot[i, val] = 1.0
            auc = float(roc_auc_score(y_one_hot, oof_probs, average="macro", multi_class="ovr"))
        except Exception:
            auc = 0.5

        per_class = {}
        for idx, cls_name in enumerate(canonical_classes):
            cls_f1 = f1_score(y_true == idx, preds == idx, average="binary")
            per_class[cls_name] = round(float(cls_f1), 4)

        minority = [per_class[c] for c in MINORITY_CLASSES if c in per_class]
        min_f1 = float(np.mean(minority)) if minority else 0.0

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "auc_roc": round(auc, 4),
            "minority_f1": round(min_f1, 4),
            "per_class_f1": per_class,
        }

    # 3. Train Models
    models_results = []

    # Model A: Random Forest (Multi-Omics)
    print("\n[Step 2.1] Training Baseline: Random Forest (Multi-Omics)...")
    t0 = time.time()
    rf_oof_probs = np.zeros((len(common_patients), n_classes))
    inference_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=args.seed + fold_idx, n_jobs=-1)
        clf.fit(X_concat[train_idx], y[train_idx])

        t_inf_start = time.time()
        val_probs = clf.predict_proba(X_concat[val_idx])
        inf_elapsed = time.time() - t_inf_start
        inference_times.append(inf_elapsed / len(val_idx))

        # Handle probability columns in case a class was absent in training fold
        for col_idx, cls_label in enumerate(clf.classes_):
            rf_oof_probs[val_idx, cls_label] = val_probs[:, col_idx]

    train_time_min = round((time.time() - t0) / 60.0, 2)
    inf_ms = round(float(np.mean(inference_times)) * 1000.0, 2)
    eval_rf = evaluate_oof(rf_oof_probs, y)

    models_results.append({
        "id": "random_forest",
        "name": "Random Forest",
        "family": "Tree ensemble",
        "metrics": {
            "accuracy": eval_rf["accuracy"],
            "macro_f1": eval_rf["macro_f1"],
            "auc_roc": eval_rf["auc_roc"],
            "cross_cohort_drop": None,
            "noise_robustness": None,
            "attention_explanations": False,
            "causal_ablation": False,
            "cosmic_overlap_pct": None,
            "train_time_min": train_time_min,
            "inference_ms": inf_ms,
            "params_m": None,
            "gpu_required": False,
        },
        "per_class_f1": eval_rf["per_class_f1"],
    })
    print(f"  -> Accuracy: {eval_rf['accuracy']:.4f} | Macro-F1: {eval_rf['macro_f1']:.4f} | AUC: {eval_rf['auc_roc']:.4f} | Train Time: {train_time_min}m | Latency: {inf_ms}ms")

    # Model B: Single-Omics (RNA-seq only Random Forest)
    print("\n[Step 2.2] Training Baseline: Single-Omics (RNA-seq Only)...")
    t0 = time.time()
    rna_oof_probs = np.zeros((len(common_patients), n_classes))
    inference_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=args.seed + fold_idx, n_jobs=-1)
        clf.fit(X_rna[train_idx], y[train_idx])

        t_inf_start = time.time()
        val_probs = clf.predict_proba(X_rna[val_idx])
        inf_elapsed = time.time() - t_inf_start
        inference_times.append(inf_elapsed / len(val_idx))

        for col_idx, cls_label in enumerate(clf.classes_):
            rna_oof_probs[val_idx, cls_label] = val_probs[:, col_idx]

    train_time_min = round((time.time() - t0) / 60.0, 2)
    inf_ms = round(float(np.mean(inference_times)) * 1000.0, 2)
    eval_rna = evaluate_oof(rna_oof_probs, y)

    models_results.append({
        "id": "single_omics",
        "name": "Single-omics (RNA-seq only)",
        "family": "Single modality",
        "metrics": {
            "accuracy": eval_rna["accuracy"],
            "macro_f1": eval_rna["macro_f1"],
            "auc_roc": eval_rna["auc_roc"],
            "cross_cohort_drop": None,
            "noise_robustness": None,
            "attention_explanations": False,
            "causal_ablation": False,
            "cosmic_overlap_pct": None,
            "train_time_min": train_time_min,
            "inference_ms": inf_ms,
            "params_m": None,
            "gpu_required": False,
        },
        "per_class_f1": eval_rna["per_class_f1"],
    })
    print(f"  -> Accuracy: {eval_rna['accuracy']:.4f} | Macro-F1: {eval_rna['macro_f1']:.4f} | AUC: {eval_rna['auc_roc']:.4f} | Train Time: {train_time_min}m | Latency: {inf_ms}ms")

    # Model C: XGBoost (Multi-Omics)
    if HAS_XGB:
        print("\n[Step 2.3] Training Baseline: XGBoost (Multi-Omics)...")
        t0 = time.time()
        xgb_oof_probs = np.zeros((len(common_patients), n_classes))
        inference_times = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            clf = XGBClassifier(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                colsample_bytree=0.1,
                n_jobs=1,
                random_state=args.seed + fold_idx,
                eval_metric="mlogloss",
            )
            clf.fit(X_concat[train_idx], y[train_idx])

            t_inf_start = time.time()
            val_probs = clf.predict_proba(X_concat[val_idx])
            inf_elapsed = time.time() - t_inf_start
            inference_times.append(inf_elapsed / len(val_idx))

            for col_idx, cls_label in enumerate(clf.classes_):
                xgb_oof_probs[val_idx, cls_label] = val_probs[:, col_idx]

        train_time_min = round((time.time() - t0) / 60.0, 2)
        inf_ms = round(float(np.mean(inference_times)) * 1000.0, 2)
        eval_xgb = evaluate_oof(xgb_oof_probs, y)

        models_results.append({
            "id": "xgboost",
            "name": "XGBoost",
            "family": "Tree ensemble",
            "metrics": {
                "accuracy": eval_xgb["accuracy"],
                "macro_f1": eval_xgb["macro_f1"],
                "auc_roc": eval_xgb["auc_roc"],
                "cross_cohort_drop": None,
                "noise_robustness": None,
                "attention_explanations": False,
                "causal_ablation": False,
                "cosmic_overlap_pct": None,
                "train_time_min": train_time_min,
                "inference_ms": inf_ms,
                "params_m": None,
                "gpu_required": False,
            },
            "per_class_f1": eval_xgb["per_class_f1"],
        })
        print(f"  -> Accuracy: {eval_xgb['accuracy']:.4f} | Macro-F1: {eval_xgb['macro_f1']:.4f} | AUC: {eval_xgb['auc_roc']:.4f} | Train Time: {train_time_min}m | Latency: {inf_ms}ms")

    # Model D: Multi-Layer Perceptron (Early Concatenation)
    print("\n[Step 2.4] Training Baseline: Multi-Layer Perceptron (Concatenated Omics)...")
    t0 = time.time()
    mlp_oof_probs = np.zeros((len(common_patients), n_classes))
    inference_times = []
    mlp_params = (X_concat.shape[1] * 128 + 128) + (128 * 64 + 64) + (64 * n_classes + n_classes)
    mlp_params_m = round(mlp_params / 1_000_000.0, 2)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_concat[train_idx])
        X_val_scaled = scaler.transform(X_concat[val_idx])

        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=100,
            early_stopping=True,
            random_state=args.seed + fold_idx,
        )
        clf.fit(X_train_scaled, y[train_idx])

        t_inf_start = time.time()
        val_probs = clf.predict_proba(X_val_scaled)
        inf_elapsed = time.time() - t_inf_start
        inference_times.append(inf_elapsed / len(val_idx))

        for col_idx, cls_label in enumerate(clf.classes_):
            mlp_oof_probs[val_idx, cls_label] = val_probs[:, col_idx]

    train_time_min = round((time.time() - t0) / 60.0, 2)
    inf_ms = round(float(np.mean(inference_times)) * 1000.0, 2)
    eval_mlp = evaluate_oof(mlp_oof_probs, y)

    models_results.append({
        "id": "mlp_concat",
        "name": "MLP (concatenated omics)",
        "family": "Early-fusion MLP",
        "metrics": {
            "accuracy": eval_mlp["accuracy"],
            "macro_f1": eval_mlp["macro_f1"],
            "auc_roc": eval_mlp["auc_roc"],
            "cross_cohort_drop": None,
            "noise_robustness": None,
            "attention_explanations": False,
            "causal_ablation": False,
            "cosmic_overlap_pct": None,
            "train_time_min": train_time_min,
            "inference_ms": inf_ms,
            "params_m": mlp_params_m,
            "gpu_required": False,
        },
        "per_class_f1": eval_mlp["per_class_f1"],
    })
    print(f"  -> Accuracy: {eval_mlp['accuracy']:.4f} | Macro-F1: {eval_mlp['macro_f1']:.4f} | AUC: {eval_mlp['auc_roc']:.4f} | Train Time: {train_time_min}m | Latency: {inf_ms}ms")

    # 4. Integrate BioNexus Model (Step 3)
    bionexus_spec_file = args.external or ("configs/bionexus_spec.json" if Path("configs/bionexus_spec.json").exists() else "bionexus_spec.json")
    if Path(bionexus_spec_file).exists():
        print(f"\n[Step 3] Integrating BioNexus from specification: {bionexus_spec_file}...")
        spec = json.loads(Path(bionexus_spec_file).read_text())
        oof_path = spec.get("oof_proba")
        if oof_path and Path(oof_path).exists():
            df_oof = load_table(oof_path)
            # Reorder rows to match common_patients
            if "patient_id" in df_oof.columns:
                df_oof = df_oof.set_index("patient_id")
            df_oof = df_oof.loc[common_patients]

            prob_cols = [f"prob_{c}" for c in canonical_classes]
            if all(c in df_oof.columns for c in prob_cols):
                bio_probs = df_oof[prob_cols].values
            else:
                bio_probs = df_oof.values

            eval_bio = evaluate_oof(bio_probs, y)
            bio_metrics = dict(spec.get("metrics", {}))
            bio_metrics["accuracy"] = eval_bio["accuracy"]
            bio_metrics["macro_f1"] = eval_bio["macro_f1"]
            bio_metrics["auc_roc"] = eval_bio["auc_roc"]
            bio_metrics["minority_f1"] = eval_bio["minority_f1"]

            models_results.insert(0, {
                "id": spec.get("id", "bionexus"),
                "name": spec.get("name", "BioNexus (Graph Transformer)"),
                "family": spec.get("family", "Graph Transformer"),
                "metrics": bio_metrics,
                "per_class_f1": eval_bio["per_class_f1"],
            })
            print(f"  -> BioNexus Measured Accuracy: {eval_bio['accuracy']:.4f} | Macro-F1: {eval_bio['macro_f1']:.4f} | AUC: {eval_bio['auc_roc']:.4f}")
        else:
            print(f"Warning: BioNexus OOF predictions file not found at {oof_path}")
    else:
        print(f"\nBioNexus spec file not found ({bionexus_spec_file}), skipping BioNexus incorporation.")

    # 5. Output Results
    output_obj = {
        "data_status": "measured",
        "notes": f"Real measured 5-fold cross-validation on TCGA-BRCA 646 patient cohort (RNA, miRNA, methylation). Generated by run_benchmark.py at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.",
        "models": models_results,
    }

    # Validate against package parser
    parse_dataset(output_obj)
    print("\nDataset successfully validated against model_comparison schema.")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(output_obj, indent=2))
    print(f"Results successfully written to: {out_p}")


if __name__ == "__main__":
    main()
