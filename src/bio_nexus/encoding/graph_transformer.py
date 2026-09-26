"""
bio_nexus.encoding.graph_transformer
----------------------------------------
GraphGPS-based encoder: takes a single-layer patient-similarity graph
(from graph/construction.py) and produces a fixed-size embedding per
patient. Run once per omics layer (RNA, miRNA, methylation) - each
gets its own encoder instance, since the input feature dimensionality
differs per layer (60660 genes vs 1881 miRNAs vs 3957 probes).

GraphGPS = hybrid layer combining local message-passing (a GCN/GINE-style
conv) with global multi-head self-attention, in the same layer - lets
the model use both immediate-neighbor structure and long-range
attention over the whole patient graph.

Output: one embedding vector of size `embed_dim` per patient, per layer
- ready to be stacked across the 3 layers and passed into
fusion/cross_attention.py.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINEConv, GPSConv
except ImportError as e:
    raise ImportError(
        "torch_geometric.nn.GPSConv not found. This requires a reasonably "
        "recent PyTorch Geometric version (GPSConv was added in PyG 2.3+). "
        'Check your version with: python3 -c "import torch_geometric; '
        'print(torch_geometric.__version__)" and upgrade with '
        "`pip install --upgrade torch_geometric` if needed."
    ) from e


@dataclass
class GraphTransformerConfig:
    embed_dim: int = 64  # per-patient embedding size (output)
    hidden_dim: int = 64  # internal hidden size (kept equal to embed_dim by default)
    num_layers: int = 3  # number of stacked GPSConv layers
    num_heads: int = 4  # attention heads in the global-attention part of GPSConv
    dropout: float = 0.3  # higher dropout given small cohort (646 patients) - guards overfitting
    pool: str = "none"  # "none" since nodes ARE patients already - no pooling needed


class GraphGPSEncoder(nn.Module):
    """Encodes a single-layer patient-similarity graph into per-patient
    embeddings. Input: node features [num_patients, num_raw_features].
    Output: node embeddings [num_patients, embed_dim] - one row per patient."""

    def __init__(self, in_features: int, cfg: GraphTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Project raw omics features (e.g. 60660-dim RNA vector) down to hidden_dim
        # BEFORE the graph layers - GPSConv layers operate on hidden_dim throughout.
        self.input_proj = nn.Linear(in_features, cfg.hidden_dim)
        self.input_norm = nn.LayerNorm(cfg.hidden_dim)

        self.gps_layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            # local message-passing component: a simple 2-layer MLP-based edge-aware conv (GINE)
            local_mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            local_conv = GINEConv(
                local_mlp, edge_dim=1
            )  # edge_dim=1: our edge_weight is scalar similarity

            gps_layer = GPSConv(
                channels=cfg.hidden_dim,
                conv=local_conv,
                heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
            self.gps_layers.append(gps_layer)

        self.output_proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        self.output_norm = nn.LayerNorm(
            cfg.embed_dim
        )  # ensures comparable scale across modalities before fusion pooling
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """
        x: [num_patients, in_features] raw omics feature matrix
        edge_index: [2, num_edges]
        edge_weight: [num_edges] similarity scores, used as scalar edge attributes
        Returns: [num_patients, embed_dim] per-patient embeddings
        """
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        edge_attr = edge_weight.unsqueeze(-1)  # [num_edges] -> [num_edges, 1] for GINEConv

        for gps_layer in self.gps_layers:
            h = gps_layer(h, edge_index, edge_attr=edge_attr)

        embeddings = self.output_proj(h)
        embeddings = self.output_norm(embeddings)
        return embeddings


def encode_layer_graph(
    graph, cfg: GraphTransformerConfig | None = None
) -> tuple[torch.Tensor, GraphGPSEncoder]:
    """Convenience wrapper: build an encoder sized to the graph's input
    feature dimension, and produce embeddings in one call.
    `graph` is a torch_geometric.data.Data object (from graph/construction.py),
    or the raw dict fallback if torch_geometric wasn't available at save time."""
    cfg = cfg or GraphTransformerConfig()

    if hasattr(graph, "x"):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
    else:
        x, edge_index, edge_attr = graph["x"], graph["edge_index"], graph["edge_weight"]

    encoder = GraphGPSEncoder(in_features=x.shape[1], cfg=cfg)
    encoder.eval()  # NOTE: this is untrained - see note below
    with torch.no_grad():
        embeddings = encoder(x, edge_index, edge_attr)
    return embeddings, encoder


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[3]
    GRAPH_DIR = BASE_DIR / "data" / "processed" / "graphs"

    cfg = GraphTransformerConfig()

    for layer_name in ["rna", "mirna", "meth"]:
        graph = torch.load(GRAPH_DIR / f"{layer_name}_graph.pt", weights_only=False)
        embeddings, encoder = encode_layer_graph(graph, cfg)
        print(
            f"[{layer_name}] embeddings shape: {tuple(embeddings.shape)} "
            f"(patients x embed_dim={cfg.embed_dim})"
        )
        print(
            f"[{layer_name}] NOTE: encoder is UNTRAINED at this point - these embeddings "
            f"are random-initialized output, not meaningful until trained end-to-end "
            f"with the classifier (see models/classifier.py + train.py)"
        )
