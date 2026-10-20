"""Load precomputed serving artifacts from data/serving/.
Zero random/synthetic generators.
Strictly traces to real BioNexus artifacts.

Memory-optimised for Render free tier (512 MB RAM):
  - Only lightweight JSON/CSV metadata is loaded at startup (~5 MB).
  - All heavy parquet matrices, edges, features, embeddings and IG
    attributions are lazy-loaded on first access via LazyParquet / LazyDict.
"""

from __future__ import annotations

import gc
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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


# ---------------------------------------------------------------------------
# Lazy-loading helpers – keeps startup RAM < 50 MB
# ---------------------------------------------------------------------------

class LazyParquet:
    """Reads a parquet file on first access and caches the DataFrame."""
    def __init__(self, path: Path, *, dtype: np.dtype | None = np.float32):
        self._path = path
        self._dtype = dtype
        self._df: pd.DataFrame | None = None

    def get(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_parquet(self._path)
            if self._dtype is not None:
                try:
                    df = df.astype(self._dtype)
                except (ValueError, TypeError):
                    pass
            self._df = df
        return self._df

    def loaded(self) -> bool:
        return self._df is not None


class LazyDict(dict):
    """Dict whose values are LazyParquet / LazyCSV; transparently loads on __getitem__."""
    def __getitem__(self, key: str) -> pd.DataFrame:
        val = super().__getitem__(key)
        if isinstance(val, LazyParquet):
            df = val.get()
            super().__setitem__(key, df)       # cache so next access is instant
            return df
        return val

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key)


class LazyCSV:
    def __init__(self, path: Path):
        self._path = path
        self._df: pd.DataFrame | None = None
    def get(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(self._path)
        return self._df
    def loaded(self) -> bool:
        return self._df is not None


class LazyCSVDict(dict):
    def __getitem__(self, key: str) -> pd.DataFrame:
        val = super().__getitem__(key)
        if isinstance(val, LazyCSV):
            df = val.get()
            super().__setitem__(key, df)
            return df
        return val


class LazyZscoreDict(dict):
    """Computes z-score matrices on demand from the raw matrices dict."""
    def __init__(self, matrices: LazyDict, directory: Path):
        super().__init__()
        self._matrices = matrices
        self._directory = directory

    def __contains__(self, key: object) -> bool:
        return key in self._matrices

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self._matrices:
            raise KeyError(key)
        if key not in self.keys():
            z_path = self._directory / f"{key}_matrix_zscore.parquet"
            if z_path.exists():
                super().__setitem__(key, pd.read_parquet(z_path).astype(np.float32))
            else:
                m = self._matrices[key]
                computed = ((m - m.mean(axis=0)) / m.std(axis=0).replace(0, 1.0)).astype(np.float32)
                super().__setitem__(key, computed)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        return default


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------

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

    # --- Live PyTorch models & graph cache (lazy loaded) ---
    _pyg_graphs: list[Any] = field(default_factory=list, repr=False)
    fold_models: dict[int, Any] = field(default_factory=dict)
    best_model: Any | None = None
    model_loaded: bool = False
    model_source: str | None = None
    model_version: str | None = None
    model_load_error: str | None = None
    model_uri: str | None = None

    @property
    def pyg_graphs(self) -> list[Any]:
        """Lazy property for PyG graphs: loaded only on first demand."""
        if not self._pyg_graphs:
            self.ensure_pyg_graphs()
        return self._pyg_graphs

    @pyg_graphs.setter
    def pyg_graphs(self, value: list[Any]) -> None:
        self._pyg_graphs = value

    def ensure_pyg_graphs(self) -> None:
        """Lazily load PyG reference graphs if needed."""
        if self._pyg_graphs:
            return
        repo_root = self.directory.parents[1]
        graph_dir = repo_root / "data" / "processed" / "graphs"
        if (graph_dir / "rna_graph.pt").exists():
            try:
                rna_g = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
                mirna_g = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
                meth_g = torch.load(graph_dir / "meth_graph.pt", weights_only=False)
                self._pyg_graphs = [rna_g, mirna_g, meth_g]
                print(f"[serving] Lazily loaded PyG graphs into memory.")
            except Exception as e:
                print(f"[serving] Warning loading reference graphs: {e}")

    # --- Lazy IG accessor ---
    _lazy_ig: Any | None = field(default=None, repr=False)

    @property
    def ig_data(self) -> pd.DataFrame | None:
        """Lazy accessor for ig attributions."""
        if self.ig is not None:
            return self.ig
        if self._lazy_ig is not None and isinstance(self._lazy_ig, LazyParquet):
            if self._lazy_ig._path.exists():
                self.ig = self._lazy_ig.get()
                return self.ig
        return None

    @classmethod
    def load(cls, directory: Path, preload_models: bool = True) -> ArtifactStore:
        # --- Lightweight metadata (< 5 MB total) ---
        patients = pd.read_csv(directory / "patients.csv")
        oof = pd.read_parquet(directory / "oof_predictions.parquet")
        labels = json.loads((directory / "labels.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        model_card = json.loads((directory / "model_card.json").read_text())
        graph_stats = json.loads((directory / "graph_stats.json").read_text())

        manifest = {}
        if (directory / "manifest.json").exists():
            manifest = json.loads((directory / "manifest.json").read_text())

        # --- Heavy data: ALL lazy-loaded ---
        edges = LazyDict({
            layer: LazyParquet(directory / f"{layer}_graph_edges.parquet", dtype=None)
            for layer in ("rna", "mirna", "meth")
        })
        matrices = LazyDict({
            layer: LazyParquet(directory / f"{layer}_matrix.parquet", dtype=np.float32)
            for layer in ("rna", "mirna", "meth")
        })
        zscore_matrices = LazyZscoreDict(matrices, directory)

        features = LazyCSVDict({
            layer: LazyCSV(directory / f"{layer}_features.csv")
            for layer in ("rna", "mirna", "meth")
        })

        # Embeddings (small, ~89 KB) – load eagerly, needed by dashboard
        embeddings_2d = None
        if (directory / "embeddings_2d.parquet").exists():
            embeddings_2d = pd.read_parquet(directory / "embeddings_2d.parquet")

        survival = None
        if (directory / "survival.json").exists():
            survival = json.loads((directory / "survival.json").read_text())

        # IG attributions – lazy
        lazy_ig = None
        ig_path = directory / "ig_attributions.parquet"
        if ig_path.exists():
            lazy_ig = LazyParquet(ig_path, dtype=None)

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
            ig=None,
            causal=causal,
            fusion_diagnosis=fusion,
            patient_index=idx,
            _lazy_ig=lazy_ig,
        )

        # Clinical (small CSV, ~92 KB)
        clin_path = directory / "patient_clinical.csv"
        if clin_path.exists():
            store.clinical = pd.read_csv(clin_path)
            store.patient_clinical_index = {
                str(row["patient_id"]): row.to_dict()
                for _, row in store.clinical.iterrows()
            }

        # Graph explorer – lazy (don't build neighbor maps at startup)
        graphs_dir = directory / "graphs"
        if graphs_dir.is_dir():
            store._load_explorer_metadata(graphs_dir)

        # Preload PyTorch models & graphs for live endpoints
        if preload_models:
            store._load_live_models_and_graphs()

        gc.collect()
        return store

    def _load_explorer_metadata(self, graphs_dir: Path) -> None:
        """Load only lightweight explorer metadata; neighbor maps are built lazily."""
        stats_path = graphs_dir / "graph_stats.json"
        if stats_path.exists():
            self.explorer_graph_stats = json.loads(stats_path.read_text())

        node_index_path = graphs_dir / "node_index.csv"
        if node_index_path.exists():
            self.explorer_node_index = pd.read_csv(node_index_path)

        overlap_path = graphs_dir / "neighbor_overlap.parquet"
        if overlap_path.exists():
            self.explorer_overlap = pd.read_parquet(overlap_path)

        # Store paths for lazy edge loading
        self._explorer_graphs_dir = graphs_dir

    def ensure_explorer_edges(self, layer: str) -> None:
        """Lazily load explorer edges and build neighbor map for a specific layer."""
        if layer in self.explorer_neighbor_map:
            return
        graphs_dir = getattr(self, "_explorer_graphs_dir", None)
        if graphs_dir is None:
            return
        edge_path = graphs_dir / f"{layer}_edges.parquet"
        if not edge_path.exists():
            return
        df = pd.read_parquet(edge_path)
        self.explorer_edges[layer] = df
        nbr_map: dict[int, list[tuple[int, float, bool]]] = defaultdict(list)
        srcs = df["src_idx"].to_numpy(dtype=int)
        dsts = df["dst_idx"].to_numpy(dtype=int)
        sims = df["cosine_similarity"].to_numpy(dtype=float)
        if "directed_knn_flag" in df.columns:
            flags = df["directed_knn_flag"].to_numpy(dtype=bool)
        elif "is_direct_knn" in df.columns:
            flags = df["is_direct_knn"].to_numpy(dtype=bool)
        else:
            flags = np.zeros(len(df), dtype=bool)
        for s, d, c, f in zip(srcs, dsts, sims, flags):
            nbr_map[s].append((d, float(c), bool(f)))
        self.explorer_neighbor_map[layer] = dict(nbr_map)

    def _load_live_models_and_graphs(self) -> None:
        """Attempt to load Production/champion model directly or from MLflow.
        PyG graphs are NOT loaded here; they remain lazy-loaded on demand to stay under 512MB RAM.
        """
        repo_root = self.directory.parents[1]
        ckpt_dir = repo_root / "data" / "processed" / "checkpoints"

        # 1. Attempt to load registered model
        model_uri = os.getenv("BIONEXUS_MODEL_URI", "models:/mosaic-subtype-classifier/Production")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            db_path = repo_root / "mlflow.db"
            tracking_uri = f"sqlite:///{db_path.resolve()}" if db_path.exists() else "sqlite:///:memory:"
        self.model_uri = model_uri
        self.model_loaded = False
        self.best_model = None

        # FAST PATH: Try direct torch.load of model.pth to avoid MLflow package bloat in memory
        direct_candidates = [
            repo_root / "mlruns" / "3" / "fe80245bc750408c9974c3ddcebab644" / "artifacts" / "model" / "data" / "model.pth",
            Path("/mlruns") / "3" / "fe80245bc750408c9974c3ddcebab644" / "artifacts" / "model" / "data" / "model.pth",
        ]
        mlruns_dir = repo_root / "mlruns"
        if mlruns_dir.exists():
            for pth in mlruns_dir.rglob("model.pth"):
                if pth not in direct_candidates:
                    direct_candidates.append(pth)

        for cand in direct_candidates:
            if cand.exists():
                try:
                    loaded = torch.load(cand, map_location="cpu", weights_only=False)
                    if isinstance(loaded, torch.nn.Module):
                        loaded.eval()
                        self.best_model = loaded
                        self.model_loaded = True
                        self.model_source = "direct_artifact_model_pth"
                        self.model_version = model_uri
                        self.model_load_error = None
                        print(f"[serving] Successfully loaded champion model directly from {cand}")
                        gc.collect()
                        return
                except Exception as ex:
                    print(f"[serving] Direct load attempt failed for {cand}: {ex}")

        # MLFLOW PATH: If direct load not found, try MLflow registry loader
        try:
            import mlflow
            import mlflow.pytorch
            mlflow.set_tracking_uri(tracking_uri)
            try:
                loaded_model = mlflow.pytorch.load_model(model_uri)
            except Exception as direct_exc:
                loaded_model = None
                try:
                    from mlflow.client import MlflowClient
                    client = MlflowClient(tracking_uri=tracking_uri)
                    parts = model_uri.replace("models:/", "").split("/")
                    m_name = parts[0]
                    stage_or_ver = parts[1] if len(parts) > 1 else "Production"
                    mvs = client.get_latest_versions(m_name, stages=[stage_or_ver])
                    if mvs:
                        raw_src = mvs[0].source.replace("file:", "").replace("\\", "/")
                        if "mlruns/" in raw_src:
                            suffix = raw_src[raw_src.index("mlruns/") + len("mlruns/") :]
                            candidates = [
                                Path("/mlruns") / suffix,
                                repo_root / "mlruns" / suffix,
                            ]
                            for cand in candidates:
                                if cand.exists() and (cand / "MLmodel").exists():
                                    loaded_model = mlflow.pytorch.load_model(str(cand))
                                    break
                except Exception:
                    pass
                if loaded_model is None:
                    raise direct_exc

            loaded_model.eval()
            self.best_model = loaded_model
            self.model_loaded = True
            self.model_source = "mlflow_registry"
            self.model_version = model_uri
            self.model_load_error = None
            print(f"[serving] Successfully loaded champion model from MLflow: {model_uri}")
        except Exception as mlflow_exc:
            err_msg = f"Failed to load from MLflow ({model_uri}): {mlflow_exc}"
            self.model_load_error = err_msg
            print(f"[serving] {err_msg}")

            # Optional local checkpoint fallback
            b_path = ckpt_dir / "best_model.pt"
            if b_path.exists():
                try:
                    from bio_nexus.models.classifier import BioNexusClassifier
                    bm = BioNexusClassifier(
                        rna_in_features=500,
                        mirna_in_features=100,
                        meth_in_features=500,
                    )
                    bm.load_state_dict(torch.load(b_path, map_location="cpu", weights_only=True))
                    bm.eval()
                    self.best_model = bm
                    self.model_loaded = True
                    self.model_source = "local_checkpoint_provisional"
                    self.model_version = "provisional_best_model"
                    self.model_load_error = None
                    print(f"[serving] Loaded provisional local checkpoint: {b_path}")
                except Exception as fb_exc:
                    self.model_load_error = f"{err_msg}; local fallback failed: {fb_exc}"
                    print(f"[serving] Fallback failed: {fb_exc}")

        gc.collect()

    def versions(self) -> dict[str, str]:
        out = {}
        for p in self.directory.iterdir():
            if p.is_file():
                out[p.name] = str(int(p.stat().st_mtime))
        return out
