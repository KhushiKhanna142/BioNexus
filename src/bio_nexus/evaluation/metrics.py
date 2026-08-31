"""
Evaluation metrics — accuracy, macro-F1, AUC-ROC, cross-cohort accuracy drop.

Owner: shared (MLOps wires this into the training loop + MLflow logging;
ML side defines what "correct" means for edge cases like class imbalance).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc_roc": roc_auc_score(y_true, y_proba, multi_class="ovr"),
    }


def cross_cohort_accuracy_drop(train_cohort_acc: float, held_out_cohort_acc: float) -> float:
    """Positive value = model degrades on unseen cohort (expected direction)."""
    return train_cohort_acc - held_out_cohort_acc
