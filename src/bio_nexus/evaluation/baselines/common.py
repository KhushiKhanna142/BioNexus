"""
bio_nexus.evaluation.baselines.common
-------------------------------------
Shared plumbing for the baseline comparison: identical outer folds for every
method, two evaluation protocols, metrics, and result (de)serialisation.

PROTOCOLS (this is the part reviewers will look at):

  inner_val  (USE THIS FOR THE PAPER)
      Outer fold k is a true held-out test set. From the outer-training nodes we
      carve a stratified inner validation set (default 20%). Gradient updates /
      model fitting use only `fit_idx`; early stopping / model selection uses only
      `stop_idx`; `test_idx` labels are never touched until the final prediction.

  eval_fold  (ONLY to reproduce the numbers train.py already reports)
      Early stopping is done on the same fold that is reported (this is what
      train.py does). It slightly flatters every method that early-stops, so do
      not put these numbers in a results table without saying so.

PARITY RULE: every method trains on exactly the same labeled nodes (`fit_idx`).
Methods with a fixed schedule (MOGONET, sklearn) simply do not use `stop_idx`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from bio_nexus.models.classifier import PAM50_SUBTYPES

N_CLASSES = len(PAM50_SUBTYPES)
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def get_device() -> torch.device:
    # Same rule as train.py. MPS is deliberately NOT used: GPSConv/scatter ops have
    # patchy MPS support, and results must be comparable with your existing runs.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataBundle:
    """Everything load_data() returns, plus helpers."""

    rna_graph: Any
    mirna_graph: Any
    meth_graph: Any
    y: np.ndarray
    node_order: list

    @property
    def graphs(self):
        return (self.rna_graph, self.mirna_graph, self.meth_graph)

    def layer_arrays(self) -> list[np.ndarray]:
        """Raw per-layer feature matrices [n_patients, n_features], order: rna, mirna, meth."""
        return [g.x.detach().cpu().numpy().astype(np.float32) for g in self.graphs]


@dataclass
class FoldTask:
    seed: int
    fold: int
    protocol: str
    y: np.ndarray  # labels of ALL nodes; only fit/stop labels may be used for training/selection
    fit_idx: np.ndarray  # labels used for parameter updates / model fitting
    stop_idx: np.ndarray  # labels used ONLY for early stopping / model selection
    test_idx: np.ndarray  # outer evaluation fold


def outer_folds(y: np.ndarray, n_folds: int, seed: int):
    """Identical call to train.py's split, so seed=42 reproduces its fold membership."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def make_task(
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    fold: int,
    protocol: str,
    inner_val_frac: float,
) -> FoldTask:
    if protocol == "eval_fold":
        fit_idx, stop_idx = np.sort(train_idx), np.sort(test_idx)
    elif protocol == "inner_val":
        fit_idx, stop_idx = train_test_split(
            train_idx,
            test_size=inner_val_frac,
            stratify=y[train_idx],
            random_state=seed * 1000 + fold,
        )
        fit_idx, stop_idx = np.sort(fit_idx), np.sort(stop_idx)
    else:
        raise ValueError(f"unknown protocol: {protocol}")
    return FoldTask(seed, fold, protocol, y, fit_idx, stop_idx, np.sort(test_idx))


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def balanced_class_weights(y_fit: np.ndarray) -> np.ndarray:
    """Same weighting as train.py (sklearn 'balanced', from training labels only)."""
    return compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=y_fit)


# ----------------------------------------------------------------------------
# result IO
# ----------------------------------------------------------------------------
def result_path(root: Path, protocol: str, method: str, seed: int) -> Path:
    return root / protocol / f"{method}__seed{seed}.json"


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, cls=_NpEncoder))
