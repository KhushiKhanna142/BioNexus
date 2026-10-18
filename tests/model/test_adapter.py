"""
tests/model/test_adapter.py
---------------------------
Unit and contract tests for BioNexusModelAdapter.
Verifies model loading, contract shapes, metadata, and probability bounds.
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
from torch_geometric.data import Data

from bio_nexus.models.adapter import BioNexusModelAdapter
from bio_nexus.models.classifier import PAM50_SUBTYPES


def test_adapter_metadata_and_shapes():
    adapter = BioNexusModelAdapter.load()
    meta = adapter.get_metadata()

    assert meta["subtypes"] == PAM50_SUBTYPES
    assert meta["expected_dimensions"]["rna"] == 5000
    assert meta["expected_dimensions"]["mirna"] == 1881
    assert meta["expected_dimensions"]["meth"] == 3957

    # Run contract check with small test graph
    num_nodes = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)

    g_rna = Data(x=torch.randn(num_nodes, 5000), edge_index=edge_index, edge_attr=edge_attr)
    g_mirna = Data(x=torch.randn(num_nodes, 1881), edge_index=edge_index, edge_attr=edge_attr)
    g_meth = Data(x=torch.randn(num_nodes, 3957), edge_index=edge_index, edge_attr=edge_attr)

    probs = adapter.predict_proba(g_rna, g_mirna, g_meth)
    assert probs.shape == (num_nodes, 5)
    # Each row must sum to 1.0 within floating point precision
    assert (probs.sum(axis=1) >= 0.99).all() and (probs.sum(axis=1) <= 1.01).all()

    preds = adapter.predict(g_rna, g_mirna, g_meth)
    assert len(preds) == num_nodes
    for p in preds:
        assert p in PAM50_SUBTYPES
