"""
Typed config loading via Pydantic + OmegaConf/Hydra.

Why this exists: every hyperparameter, path, and seed must flow through
here -- never hardcoded in training scripts. This is what makes runs
reproducible and what lets MLflow log a complete, structured record of
exactly what produced a given result.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class _FrozenBase(BaseModel):
    """Every config (and every nested config) is immutable once loaded.
    Freezing only the top-level model is a common mistake -- nested
    BaseModel fields stay mutable unless they inherit this too."""

    model_config = {"frozen": True}


class DataConfig(_FrozenBase):
    tcga_project_ids: list[str] = Field(default_factory=lambda: ["TCGA-BRCA"])
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    held_out_cohort: str | None = None  # e.g. METABRIC, for cross-cohort eval
    random_seed: int = 42


class ModelConfig(_FrozenBase):
    architecture: Literal["graphormer", "gat"] = "graphormer"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1


class TrainingConfig(_FrozenBase):
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    device: Literal["cpu", "cuda"] = "cuda"
    mlflow_experiment_name: str = "bionexus-cancer-subtype"


class BioNexusConfig(_FrozenBase):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
