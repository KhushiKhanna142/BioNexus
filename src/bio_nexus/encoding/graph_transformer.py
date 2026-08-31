"""
Stage 3 — Graph Transformer Encoding.

Owner: ML side. Graphormer-style structural/positional encoding + self-attention.

MLOps contract: this module must accept `configs/model/*.yaml` params only
via a typed config object (see bio_nexus.utils.config), never via free-floating
kwargs, so W&B/MLflow can log the exact config used for every run.
"""

from __future__ import annotations

import torch
from torch import nn


class GraphTransformerEncoder(nn.Module):
    """Encodes a single omics-layer graph into node/patient embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        raise NotImplementedError("ML implementation pending — see Stage 3 in docs.")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
