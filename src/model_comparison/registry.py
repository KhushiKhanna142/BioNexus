"""
Metric registry: the single source of truth for WHAT is compared and HOW.

Adding a new comparison parameter = add one MetricDef here (+ a value in the
results file). Ranking, "leads on X of Y", excels/lags text and the API all
pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower"]
Kind = Literal["number", "boolean"]

FOCUS_MODEL_ID = "bionexus"

PAM50_CLASSES = ["LumA", "LumB", "Her2", "Basal", "Normal"]
MINORITY_CLASSES = ["Her2", "Normal"]  # rarer subtypes, hardest for baselines

CATEGORIES = [
    {"id": "performance", "label": "Predictive performance"},
    {"id": "robustness", "label": "Robustness & generalization"},
    {"id": "interpretability", "label": "Interpretability"},
    {"id": "cost", "label": "Practical cost"},
]


@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    category: str
    direction: Direction          # which way is "better"
    unit: str                     # ratio | pct | pp | min | ms | M | bool
    description: str              # plain-language tooltip for the UI
    excels_reason: str            # shown when the focus model ranks #1
    lags_reason: str              # shown when the focus model does not
    kind: Kind = "number"

    def fmt(self, v) -> str:
        if v is None:
            return "n/a"
        if self.kind == "boolean":
            return "Yes" if v else "No"
        return {
            "ratio": f"{v * 100:.1f}%",
            "pct": f"{v:.1f}%",
            "pp": f"{v:.1f} pp",
            "min": f"{v:.0f} min",
            "ms": f"{v:.0f} ms",
            "M": f"{v:.1f}M",
        }.get(self.unit, f"{v:g}")


METRICS: list[MetricDef] = [
    # ---------------------------------------------------------- performance
    MetricDef(
        "accuracy", "Accuracy", "performance", "higher", "ratio",
        "Share of patients whose PAM50 subtype is predicted correctly.",
        "Graph structure lets the model use gene-gene interactions that flat models ignore.",
        "Gap is small: all strong models saturate on the large, easy subtypes.",
    ),
    MetricDef(
        "macro_f1", "Macro F1", "performance", "higher", "ratio",
        "F1 averaged equally over the 5 subtypes, so rare subtypes count as much as common ones.",
        "Balanced across subtypes, not just good on the common ones.",
        "Rare-subtype errors pull the average down.",
    ),
    MetricDef(
        "auc_roc", "AUC-ROC", "performance", "higher", "ratio",
        "How well the model separates subtypes across all decision thresholds.",
        "Cleaner class separation in the fused multi-omics embedding.",
        "Probability calibration is comparable across the top models.",
    ),
    # ------------------------------------------------------------ robustness
    MetricDef(
        "cross_cohort_drop", "Cross-cohort accuracy drop", "robustness", "lower", "pp",
        "Accuracy lost (percentage points) when a model trained on one cohort is tested on an independent cohort. Lower is better.",
        "Biological graph priors transfer across cohorts better than purely statistical features.",
        "Another model generalizes better across cohorts in this run.",
    ),
    MetricDef(
        "noise_robustness", "Noise robustness", "robustness", "higher", "pct",
        "Percent of baseline accuracy retained when 20% of input features are corrupted.",
        "Neighbourhood aggregation smooths out noisy individual features.",
        "Another model degrades less under corrupted inputs.",
    ),
    MetricDef(
        "minority_f1", "Rare-subtype F1 (Her2 + Normal)", "robustness", "higher", "ratio",
        "Mean F1 on the two least frequent subtypes, where most models struggle.",
        "Attention over related genes helps where few training samples exist.",
        "Rare subtypes remain hard for this model.",
    ),
    # -------------------------------------------------------- interpretability
    MetricDef(
        "attention_explanations", "Attention-based explanations", "interpretability", "higher", "bool",
        "Whether the model exposes which genes/features it attended to.",
        "Attention maps show which genes the model focused on.",
        "Not available for this model.",
        kind="boolean",
    ),
    MetricDef(
        "causal_ablation", "Causal ablation validated", "interpretability", "higher", "bool",
        "Whether flagged genes were removed and the prediction change was measured, proving they matter.",
        "Goes beyond attention maps: removing flagged genes measurably changes predictions.",
        "Causal validation not performed for this model.",
        kind="boolean",
    ),
    MetricDef(
        "cosmic_overlap_pct", "Top genes in COSMIC / OncoKB", "interpretability", "higher", "pct",
        "Percent of the model's top-ranked genes that are known cancer driver genes.",
        "Its important genes line up with known cancer biology.",
        "Its top genes overlap less with known drivers.",
    ),
    # ------------------------------------------------------------------ cost
    MetricDef(
        "train_time_min", "Training time", "cost", "lower", "min",
        "Wall-clock minutes to train on the full cohort.",
        "Efficient training for a graph model.",
        "Graph encoders and attention are costlier to train than tree/linear models.",
    ),
    MetricDef(
        "inference_ms", "Inference latency", "cost", "lower", "ms",
        "Milliseconds to classify one patient.",
        "Fast enough for interactive use.",
        "Attention over graph neighbours costs more per prediction than a tree ensemble.",
    ),
    MetricDef(
        "params_m", "Parameters", "cost", "lower", "M",
        "Number of learnable parameters, in millions (tree models are not applicable).",
        "Compact for what it does.",
        "Transformer layers carry more parameters than simpler models.",
    ),
    MetricDef(
        "gpu_required", "GPU required for training", "cost", "lower", "bool",
        "Whether training realistically needs a GPU.",
        "Runs without a GPU.",
        "Needs a GPU for practical training times.",
        kind="boolean",
    ),
]

METRIC_BY_ID = {m.id: m for m in METRICS}

# Radar chart: each axis averages the normalised score (0-100) of its metrics.
RADAR_AXES = [
    {"id": "accuracy", "label": "Accuracy", "metrics": ["accuracy"], "log": False},
    {"id": "f1", "label": "Macro F1", "metrics": ["macro_f1"], "log": False},
    {"id": "robustness", "label": "Noise robustness", "metrics": ["noise_robustness"], "log": False},
    {"id": "generalization", "label": "Cross-cohort", "metrics": ["cross_cohort_drop"], "log": False},
    {"id": "interpretability", "label": "Interpretability",
     "metrics": ["attention_explanations", "causal_ablation", "cosmic_overlap_pct"], "log": False},
    {"id": "speed", "label": "Speed", "metrics": ["inference_ms"], "log": True},
    {"id": "efficiency", "label": "Efficiency", "metrics": ["params_m"], "log": True},
]
