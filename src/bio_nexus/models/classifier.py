"""
Stage 5a — Classification head.

Owner: ML side. Takes fused patient embedding -> cancer subtype logits.
"""
from __future__ import annotations
import torch
from torch import nn


class SubtypeClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_subtypes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_subtypes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
