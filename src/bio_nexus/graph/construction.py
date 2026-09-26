"""
bio_nexus.graph.construction
------------------------------
Builds a patient-similarity graph per omics layer (RNA-seq, methylation,
miRNA) from the final locked-in cohort matrices. Each graph is a
k-nearest-neighbor graph over patients, using cosine similarity between
patients' feature vectors within that layer.

Output: one torch_geometric.data.Data object per layer, saved to
data/processed/graphs/{layer}_graph.pt - ready for the encoding/
graph_transformer.py module (GraphGPS) to consume.

Design choice (log this in your methods section): k=10 nearest neighbors,
cosine similarity, undirected (edges symmetrized). These are configurable
via configs/default.yaml under `graph_construction`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphConstructionConfig:
    k_neighbors: int = 10
    metric: str = "cosine"  # "cosine" or "correlation"
    symmetric: bool = True  # symmetrize kNN edges (make undirected)
    self_loops: bool = False


def build_knn_graph(feature_matrix: pd.DataFrame, cfg: GraphConstructionConfig):
    """Build a k-NN patient-similarity graph from a patients x features matrix.

    Returns:
        edge_index: LongTensor [2, num_edges] - PyTorch Geometric format
        edge_weight: FloatTensor [num_edges] - similarity score per edge
        node_ids: list of patient IDs in row order (matches feature_matrix.index)
    """
    X = feature_matrix.to_numpy(dtype=np.float32)
    node_ids = list(feature_matrix.index)
    n = X.shape[0]

    if cfg.metric == "cosine":
        sim = cosine_similarity(X)
    elif cfg.metric == "correlation":
        sim = np.corrcoef(X)
    else:
        raise ValueError(f"Unknown metric: {cfg.metric}")

    np.fill_diagonal(sim, -1e6)  # large finite sentinel to exclude self from neighbor search
    # (sklearn's precomputed-metric NearestNeighbors rejects actual inf/-inf values)

    # For each patient, find their top-k most similar patients
    nn = NearestNeighbors(n_neighbors=cfg.k_neighbors, metric="precomputed")
    # convert similarity -> distance for sklearn's NearestNeighbors API
    dist = 1 - sim  # self-distance becomes a large finite number, never selected as nearest
    nn.fit(dist)
    _, neighbor_idx = nn.kneighbors(dist)

    src, dst, weights = [], [], []
    for i in range(n):
        for j in neighbor_idx[i]:
            src.append(i)
            dst.append(j)
            weights.append(sim[i, j])

    if cfg.symmetric:
        # add reverse edges, then de-duplicate (keep max weight if edge exists both ways)
        src2 = src + dst
        dst2 = dst + src
        w2 = weights + weights
        edge_df = pd.DataFrame({"src": src2, "dst": dst2, "w": w2})
        edge_df = edge_df.groupby(["src", "dst"], as_index=False)["w"].max()
        src, dst, weights = edge_df["src"].tolist(), edge_df["dst"].tolist(), edge_df["w"].tolist()

    if cfg.self_loops:
        for i in range(n):
            src.append(i)
            dst.append(i)
            weights.append(1.0)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight, node_ids


def build_all_layer_graphs(
    processed_final_dir: Path, out_dir: Path, cfg: GraphConstructionConfig = None
):
    """Build and save a k-NN graph for each of the 3 omics layers, using
    the final locked-in cohort matrices (rna/mirna/meth_matrix_final.csv)."""
    cfg = cfg or GraphConstructionConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = {
        "rna": processed_final_dir / "rna_matrix_final.csv",
        "mirna": processed_final_dir / "mirna_matrix_final.csv",
        "meth": processed_final_dir / "meth_matrix_final.csv",
    }

    node_id_reference = None
    for layer_name, path in layers.items():
        matrix = pd.read_csv(path, index_col=0)
        edge_index, edge_weight, node_ids = build_knn_graph(matrix, cfg)

        # sanity check: node order must match across all 3 layers for later fusion
        if node_id_reference is None:
            node_id_reference = node_ids
        elif node_ids != node_id_reference:
            raise ValueError(
                f"[{layer_name}] patient order doesn't match reference layer - "
                f"fusion requires identical node ordering across all 3 graphs"
            )

        x = torch.tensor(matrix.to_numpy(dtype=np.float32))

        try:
            from torch_geometric.data import Data

            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
            torch.save(graph, out_dir / f"{layer_name}_graph.pt")
        except ImportError:
            # fall back to saving raw tensors if torch_geometric isn't installed yet
            torch.save(
                {
                    "x": x,
                    "edge_index": edge_index,
                    "edge_weight": edge_weight,
                    "node_ids": node_ids,
                },
                out_dir / f"{layer_name}_graph.pt",
            )

        print(
            f"[{layer_name}] graph built: {x.shape[0]} nodes, {edge_index.shape[1]} edges "
            f"-> saved {out_dir / f'{layer_name}_graph.pt'}"
        )

    pd.Series(node_id_reference, name="patient_id").to_csv(
        out_dir / "graph_node_order.csv", index=False
    )
    print(f"\nAll 3 layer graphs saved to {out_dir}, node order confirmed consistent.")


if __name__ == "__main__":
    # Adjust these paths to match your repo layout
    BASE_DIR = Path(__file__).resolve().parents[3]  # repo root, from src/bio_nexus/graph/
    PROCESSED_FINAL = BASE_DIR / "data" / "processed" / "final"
    GRAPH_OUT = BASE_DIR / "data" / "processed" / "graphs"

    build_all_layer_graphs(PROCESSED_FINAL, GRAPH_OUT)
