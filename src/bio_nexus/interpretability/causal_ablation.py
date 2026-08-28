"""
Stage 5b — Causal ablation-based interpretability.

Owner: ML side, this is BioNexus's core novelty claim (Gap 1 in the doc).
Systematically masks/removes top-attended genes and measures the real
accuracy drop, then cross-checks against COSMIC/OncoKB driver-gene lists.

MLOps contract: every ablation run must log its Ablation Importance Score
to MLflow as a metric, tagged with the gene/feature ablated, so results
are queryable later (not just printed to stdout and lost).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AblationResult:
    feature: str
    baseline_accuracy: float
    ablated_accuracy: float

    @property
    def importance_score(self) -> float:
        return self.baseline_accuracy - self.ablated_accuracy


def run_ablation_sweep(model, dataloader, top_features: list[str]) -> list[AblationResult]:
    raise NotImplementedError("ML implementation pending — see Stage 5 in docs.")
