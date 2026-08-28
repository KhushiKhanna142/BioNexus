"""
Stage 2 — Graph Construction.

Owner: ML side. Builds per-omics-layer graphs from STRING DB / BioGRID
interactions or patient-similarity metrics.

MLOps contract: graph construction must be deterministic given the same
input data + config (fixed random_state everywhere). Non-deterministic
graph construction breaks experiment reproducibility and DVC caching.
"""
from __future__ import annotations
import networkx as nx
import pandas as pd


def build_gene_interaction_graph(
    expression: pd.DataFrame,
    interactions: pd.DataFrame,
    min_confidence: float = 0.7,
) -> nx.Graph:
    """Build a gene-gene graph from a PPI edge list filtered by confidence.

    Args:
        expression: genes-as-columns DataFrame, used to restrict graph
            nodes to genes actually present in the dataset.
        interactions: edge list with columns [gene_a, gene_b, confidence].
        min_confidence: STRING/BioGRID confidence threshold for keeping an edge.

    Returns:
        networkx.Graph with genes as nodes.
    """
    raise NotImplementedError("ML implementation pending — see Stage 2 in docs.")
