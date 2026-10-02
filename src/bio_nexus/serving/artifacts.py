"""Load precomputed serving artifacts from data/serving/.
Zero random/synthetic generators.
Strictly traces to real BioNexus artifacts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REQUIRED_FILES = [
    "patients.csv",
    "oof_predictions.parquet",
    "labels.json",
    "metrics.json",
    "model_card.json",
    "graph_stats.json",
    "manifest.json",
    "embeddings_2d.parquet",
    "survival.json",
    "patient_clinical.csv",
    "rna_graph_edges.parquet",
    "mirna_graph_edges.parquet",
    "meth_graph_edges.parquet",
    "rna_matrix.parquet",
    "mirna_matrix.parquet",
    "meth_matrix.parquet",
    "rna_features.csv",
    "mirna_features.csv",
    "meth_features.csv",
]

OPTIONAL_FILES = [
    "ig_attributions.parquet",
    "causal_validation.json",
    "fusion_diagnosis.json",
]

LAYER_FILE = {"rna": "rna", "mirna": "mirna", "meth": "meth", "methylation": "meth"}


def serving_dir(base_dir: Path) -> Path:
    import os
    raw = os.getenv("BIONEXUS_SERVING_DIR")
    if raw:
        return Path(raw)
    return base_dir / "data" / "serving"


def list_missing(directory: Path) -> list[str]:
    missing = []
    for name in REQUIRED_FILES:
        if not (directory / name).exists():
            missing.append(str(directory / name))
    return missing


def list_optional_missing(directory: Path) -> list[str]:
    return [str(directory / n) for n in OPTIONAL_FILES if not (directory / n).exists()]


@dataclass
class ArtifactStore:
    directory: Path
    patients: pd.DataFrame
    oof: pd.DataFrame
    labels: dict[str, Any]
    metrics: dict[str, Any]
    model_card: dict[str, Any]
    graph_stats: dict[str, Any]
    manifest: dict[str, Any]
    edges: dict[str, pd.DataFrame]
    matrices: dict[str, pd.DataFrame]
    zscore_matrices: dict[str, pd.DataFrame]
    features: dict[str, pd.DataFrame]
    embeddings_2d: pd.DataFrame | None
    survival: dict[str, Any] | None
    ig: pd.DataFrame | None
    causal: dict[str, Any] | None
    fusion_diagnosis: dict[str, Any] | None
    patient_index: dict[str, int] = field(default_factory=dict)

    # --- Clinical metadata ---
    clinical: pd.DataFrame | None = None
    patient_clinical_index: dict[str, dict[str, Any]] = field(default_factory=dict)

    # --- Graph Explorer artifacts ---
    explorer_graph_stats: dict[str, Any] | None = None
    explorer_edges: dict[str, pd.DataFrame] = field(default_factory=dict)
    explorer_neighbor_map: dict[str, dict[int, list[tuple[int, float, bool]]]] = field(
        default_factory=dict
    )
    explorer_overlap: pd.DataFrame | None = None
    explorer_node_index: pd.DataFrame | None = None

    # --- Live PyTorch models & graph cache ---
    pyg_graphs: list[Any] = field(default_factory=list)
    fold_models: dict[int, Any] = field(default_factory=dict)
    best_model: Any | None = None

    @classmethod
    def load(cls, directory: Path) -> ArtifactStore:
        patients = pd.read_csv(directory / "patients.csv")
        oof = pd.read_parquet(directory / "oof_predictions.parquet")
        labels = json.loads((directory / "labels.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        model_card = json.loads((directory / "model_card.json").read_text())
        graph_stats = json.loads((directory / "graph_stats.json").read_text())

        manifest = {}
        if (directory / "manifest.json").exists():
            manifest = json.loads((directory / "manifest.json").read_text())

        edges = {
            layer: pd.read_parquet(directory / f"{layer}_graph_edges.parquet")
            for layer in ("rna", "mirna", "meth")
        }
        matrices = {
            layer: pd.read_parquet(directory / f"{layer}_matrix.parquet")
            for layer in ("rna", "mirna", "meth")
        }
        zscore_matrices = {}
        for layer in ("rna", "mirna", "meth"):
            z_path = directory / f"{layer}_matrix_zscore.parquet"
            if z_path.exists():
                zscore_matrices[layer] = pd.read_parquet(z_path)
            else:
                # Compute on the fly if missing
                m = matrices[layer]
                zscore_matrices[layer] = (m - m.mean(axis=0)) / m.std(axis=0).replace(0, 1.0)

        features = {
            layer: pd.read_csv(directory / f"{layer}_features.csv")
            for layer in ("rna", "mirna", "meth")
        }

        embeddings_2d = None
        if (directory / "embeddings_2d.parquet").exists():
            embeddings_2d = pd.read_parquet(directory / "embeddings_2d.parquet")

        survival = None
        if (directory / "survival.json").exists():
            survival = json.loads((directory / "survival.json").read_text())

        ig = None
        ig_path = directory / "ig_attributions.parquet"
        if ig_path.exists():
            ig = pd.read_parquet(ig_path)

        causal = None
        causal_path = directory / "causal_validation.json"
        if causal_path.exists():
            causal = json.loads(causal_path.read_text())

        fusion = None
        fusion_path = directory / "fusion_diagnosis.json"
        if fusion_path.exists():
            fusion = json.loads(fusion_path.read_text())

        idx = {pid: i for i, pid in enumerate(patients["patient_id"].tolist())}

        store = cls(
            directory=directory,
            patients=patients,
            oof=oof,
            labels=labels,
            metrics=metrics,
            model_card=model_card,
            graph_stats=graph_stats,
            manifest=manifest,
            edges=edges,
            matrices=matrices,
            zscore_matrices=zscore_matrices,
            features=features,
            embeddings_2d=embeddings_2d,
            survival=survival,
            ig=ig,
            causal=causal,
            fusion_diagnosis=fusion,
            patient_index=idx,
        )

        # Clinical
        clin_path = directory / "patient_clinical.csv"
        if clin_path.exists():
            store.clinical = pd.read_csv(clin_path)
            store.patient_clinical_index = {
                str(row["patient_id"]): row.to_dict()
                for _, row in store.clinical.iterrows()
            }

        # Graph explorer
        graphs_dir = directory / "graphs"
        if graphs_dir.is_dir():
            store._load_explorer_artifacts(graphs_dir)

        # Preload PyTorch models & graphs for live endpoints
        store._load_live_models_and_graphs()

        return store

    def _load_explorer_artifacts(self, graphs_dir: Path) -> None:
        stats_path = graphs_dir / "graph_stats.json"
        if stats_path.exists():
            self.explorer_graph_stats = json.loads(stats_path.read_text())

        node_index_path = graphs_dir / "node_index.csv"
        if node_index_path.exists():
            self.explorer_node_index = pd.read_csv(node_index_path)

        overlap_path = graphs_dir / "neighbor_overlap.parquet"
        if overlap_path.exists():
            self.explorer_overlap = pd.read_parquet(overlap_path)

        for layer in ("rna", "mirna", "meth"):
            edge_path = graphs_dir / f"{layer}_edges.parquet"
            if edge_path.exists():
                df = pd.read_parquet(edge_path)
                self.explorer_edges[layer] = df
                nbr_map: dict[int, list[tuple[int, float, bool]]] = defaultdict(list)
                for _, row in df.iterrows():
                    flag = bool(row.get("directed_knn_flag", row.get("is_direct_knn", False)))
                    nbr_map[int(row["src_idx"])].append(
                        (int(row["dst_idx"]), float(row["cosine_similarity"]), flag)
                    )
                self.explorer_neighbor_map[layer] = dict(nbr_map)

    def _load_live_models_and_graphs(self) -> None:
        """Preload PyG graphs and PyTorch fold checkpoints once at startup."""
        repo_root = self.directory.parents[1]
        ckpt_dir = repo_root / "data" / "processed" / "checkpoints"
        graph_dir = repo_root / "data" / "processed" / "graphs"

        if (graph_dir / "rna_graph.pt").exists():
            try:
                from bio_nexus.models.classifier import BioNexusClassifier
                rna_g = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
                mirna_g = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
                meth_g = torch.load(graph_dir / "meth_graph.pt", weights_only=False)
                self.pyg_graphs = [rna_g, mirna_g, meth_g]

                for f in range(1, 6):
                    f_path = ckpt_dir / f"fold_{f}.pt"
                    if f_path.exists():
                        m = BioNexusClassifier(
                            rna_in_features=rna_g.x.shape[1],
                            mirna_in_features=mirna_g.x.shape[1],
                            meth_in_features=meth_g.x.shape[1],
                        )
                        m.load_state_dict(torch.load(f_path, map_location="cpu", weights_only=True))
                        m.eval()
                        self.fold_models[f] = m

                b_path = ckpt_dir / "best_model.pt"
                if b_path.exists():
                    bm = BioNexusClassifier(
                        rna_in_features=rna_g.x.shape[1],
                        mirna_in_features=mirna_g.x.shape[1],
                        meth_in_features=meth_g.x.shape[1],
                    )
                    bm.load_state_dict(torch.load(b_path, map_location="cpu", weights_only=True))
                    bm.eval()
                    self.best_model = bm

                print(f"[serving] Live PyTorch models loaded: {len(self.fold_models)} folds + best_model")
            except Exception as e:
                print(f"[serving] Warning loading live models: {e}")

    def versions(self) -> dict[str, str]:
        out = {}
        for p in self.directory.iterdir():
            if p.is_file():
                out[p.name] = str(int(p.stat().st_mtime))
        return out
