"""
Training entrypoint. This is the ONE place that wires together data,
graph construction, model, and MLflow logging. Every training run —
local, CI smoke test, or production — goes through this same path.

Run:
    python -m bio_nexus.train --config configs/training/default.yaml

Design principle: this file orchestrates; it does not contain model or
data logic itself. If you're writing algorithm code here, it belongs in
one of the stage modules instead.
"""
from __future__ import annotations

import argparse

import mlflow
from omegaconf import OmegaConf

from bio_nexus.utils.config import BioNexusConfig
from bio_nexus.utils.logging import get_logger
from bio_nexus.utils.seed import set_seed

logger = get_logger(__name__)


def load_config(path: str) -> BioNexusConfig:
    raw = OmegaConf.load(path)
    return BioNexusConfig(**OmegaConf.to_container(raw, resolve=True))


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.data.random_seed)

    mlflow.set_experiment(cfg.training.mlflow_experiment_name)
    with mlflow.start_run():
        mlflow.log_params(cfg.model_dump())
        logger.info("Starting BioNexus training run")

        # --- Stage 1: Data ingestion (bio_nexus.data.ingestion) ---
        # --- Stage 2: Graph construction (bio_nexus.graph.construction) ---
        # --- Stage 3: Graph transformer encoding (bio_nexus.encoding.graph_transformer) ---
        # --- Stage 4: Multi-omics fusion (bio_nexus.fusion.cross_attention) ---
        # --- Stage 5: Classification + causal ablation + cross-cohort eval ---
        # Each stage is implemented in its own module; wire them here once
        # the ML side has filled in the NotImplementedError stubs.

        logger.info("Training run complete (pipeline stages pending ML implementation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/default.yaml")
    args = parser.parse_args()
    main(args.config)
