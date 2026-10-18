"""
Evaluate existing checkpoints (fold_1.pt to fold_5.pt and best_model.pt)
on their held-out validation patients from data/serving/patients.csv
using shared metrics.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

from bio_nexus.models.classifier import BioNexusClassifier, PAM50_SUBTYPES


def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE) across multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def main():
    base_dir = Path(".")
    serving_dir = base_dir / "data" / "serving"
    ckpt_dir = base_dir / "data" / "processed" / "checkpoints"
    graph_dir = base_dir / "data" / "processed" / "graphs"

    print("=== EVALUATING 5-FOLD CHECKPOINTS ON HELD-OUT PATIENTS ===")
    
    # 1. Load patient metadata & splits
    pats = pd.read_csv(serving_dir / "patients.csv")
    label_map = {name: i for i, name in enumerate(PAM50_SUBTYPES)}
    y_true_all = pats["true_pam50"].map(label_map).to_numpy()
    fold_ids = pats["fold_id"].to_numpy()

    # 2. Load graphs
    rna_g = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
    mirna_g = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
    meth_g = torch.load(graph_dir / "meth_graph.pt", weights_only=False)

    fold_results = {}
    oof_predictions = np.zeros(len(y_true_all), dtype=int)
    oof_probabilities = np.zeros((len(y_true_all), len(PAM50_SUBTYPES)), dtype=float)

    for fold in range(1, 6):
        ckpt_path = ckpt_dir / f"fold_{fold}.pt"
        if not ckpt_path.exists():
            print(f"[-] Fold {fold} checkpoint not found: {ckpt_path}")
            continue

        model = BioNexusClassifier(
            rna_in_features=rna_g.x.shape[1],
            mirna_in_features=mirna_g.x.shape[1],
            meth_in_features=meth_g.x.shape[1],
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.eval()

        val_mask = (fold_ids == fold)
        val_indices = np.where(val_mask)[0]
        y_val = y_true_all[val_indices]

        with torch.no_grad():
            logits, _, _, _ = model(rna_g, mirna_g, meth_g)
            val_logits = logits[val_indices]
            val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            val_preds = np.argmax(val_probs, axis=-1)

        oof_predictions[val_indices] = val_preds
        oof_probabilities[val_indices] = val_probs

        acc = accuracy_score(y_val, val_preds)
        bal_acc = balanced_accuracy_score(y_val, val_preds)
        macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)
        per_class_f1 = f1_score(y_val, val_preds, average=None, labels=list(range(len(PAM50_SUBTYPES))), zero_division=0)
        ll = log_loss(y_val, val_probs, labels=list(range(len(PAM50_SUBTYPES))))
        ece = compute_ece(val_probs, y_val)

        fold_results[f"fold_{fold}"] = {
            "val_samples": len(val_indices),
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "log_loss": round(float(ll), 4),
            "ece": round(float(ece), 4),
            "per_subtype_f1": {
                PAM50_SUBTYPES[i]: round(float(per_class_f1[i]), 4)
                for i in range(len(PAM50_SUBTYPES))
            },
        }

        print(f"\n[*] Fold {fold} (n={len(val_indices)}):")
        print(f"    Accuracy:          {acc:.4f}")
        print(f"    Balanced Accuracy: {bal_acc:.4f}")
        print(f"    Macro F1:          {macro_f1:.4f}")
        print(f"    Weighted F1:       {weighted_f1:.4f}")
        print(f"    Log Loss:          {ll:.4f}")
        print(f"    ECE:               {ece:.4f}")
        print(f"    Subtype F1:        {fold_results[f'fold_{fold}']['per_subtype_f1']}")

    # Out-of-fold Aggregate Metrics (True Cross-Validation)
    oof_acc = accuracy_score(y_true_all, oof_predictions)
    oof_bal_acc = balanced_accuracy_score(y_true_all, oof_predictions)
    oof_macro_f1 = f1_score(y_true_all, oof_predictions, average="macro", zero_division=0)
    oof_weighted_f1 = f1_score(y_true_all, oof_predictions, average="weighted", zero_division=0)
    oof_class_f1 = f1_score(y_true_all, oof_predictions, average=None, labels=list(range(len(PAM50_SUBTYPES))), zero_division=0)
    oof_ece = compute_ece(oof_probabilities, y_true_all)
    oof_ll = log_loss(y_true_all, oof_probabilities, labels=list(range(len(PAM50_SUBTYPES))))

    print("\n==========================================================")
    print("   5-FOLD OUT-OF-FOLD (OOF) CROSS-VALIDATION SUMMARY")
    print("==========================================================")
    print(f"Total Cohort Size:      {len(y_true_all)} TCGA-BRCA patients")
    print(f"OOF Accuracy:           {oof_acc:.4f}")
    print(f"OOF Balanced Accuracy:  {oof_bal_acc:.4f}")
    print(f"OOF Macro F1:           {oof_macro_f1:.4f}")
    print(f"OOF Weighted F1:        {oof_weighted_f1:.4f}")
    print(f"OOF Log Loss:           {oof_ll:.4f}")
    print(f"OOF ECE (Calibration):  {oof_ece:.4f}")
    print(f"Per-Subtype F1 Scores:")
    for i, sub in enumerate(PAM50_SUBTYPES):
        print(f"  {sub:8s}: {oof_class_f1[i]:.4f}")

    # Inspect best_model.pt across entire cohort and on its best fold
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        bm = BioNexusClassifier(
            rna_in_features=rna_g.x.shape[1],
            mirna_in_features=mirna_g.x.shape[1],
            meth_in_features=meth_g.x.shape[1],
        )
        bm.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        bm.eval()

        with torch.no_grad():
            bm_logits, _, _, _ = bm(rna_g, mirna_g, meth_g)
            bm_probs = F.softmax(bm_logits, dim=-1).cpu().numpy()
            bm_preds = np.argmax(bm_probs, axis=-1)

        print("\n==========================================================")
        print("   best_model.pt DIAGNOSTIC EVALUATION")
        print("==========================================================")
        # Check against each fold's validation set to identify which fold it was saved from
        best_fold_idx = None
        best_fold_match_score = -1.0
        for fold in range(1, 6):
            val_idx = np.where(fold_ids == fold)[0]
            val_f1 = f1_score(y_true_all[val_idx], bm_preds[val_idx], average="macro", zero_division=0)
            print(f"  Performance on Fold {fold} validation slice: Macro F1 = {val_f1:.4f}")
            if val_f1 > best_fold_match_score:
                best_fold_match_score = val_f1
                best_fold_idx = fold

        full_acc = accuracy_score(y_true_all, bm_preds)
        full_macro = f1_score(y_true_all, bm_preds, average="macro", zero_division=0)
        print(f"\nSelection Origin: best_model.pt corresponds to Fold {best_fold_idx} checkpoint (Macro F1 = {best_fold_match_score:.4f} on its own fold).")
        print(f"When evaluated on the full 646 cohort: Accuracy={full_acc:.4f}, Macro F1={full_macro:.4f}.")
        print("Note: Its single-fold score is optimistic due to fold selection bias. The honest research metric is the 5-fold OOF Macro F1 above.")

    summary_payload = {
        "status": "provisional_plumbing_checkpoints",
        "data_validated": False,
        "cohort_size": len(y_true_all),
        "n_folds": 5,
        "oof_metrics": {
            "accuracy": round(float(oof_acc), 4),
            "balanced_accuracy": round(float(oof_bal_acc), 4),
            "macro_f1": round(float(oof_macro_f1), 4),
            "weighted_f1": round(float(oof_weighted_f1), 4),
            "log_loss": round(float(oof_ll), 4),
            "ece": round(float(oof_ece), 4),
            "per_subtype_f1": {PAM50_SUBTYPES[i]: round(float(oof_class_f1[i]), 4) for i in range(len(PAM50_SUBTYPES))},
        },
        "per_fold": fold_results,
    }
    
    rep_path = base_dir / "reports" / "checkpoint_evaluation.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(summary_payload, indent=2))
    print(f"\nSaved evaluation report to {rep_path}")

if __name__ == "__main__":
    main()
