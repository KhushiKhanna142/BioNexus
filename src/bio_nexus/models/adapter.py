"""
src/bio_nexus/models/adapter.py
--------------------------------
Production Model Adapter and Contract for BioNexus / MOSAIC Subtype Classifier.

Contract:
- Input: Modality feature vectors (dict, numpy array, PyTorch Tensor, or DataFrame)
- Output: PAM50 subtype probabilities (Basal, Her2, LumA, LumB, Normal)
- Metadata: Version, stage, architecture, features expected, and evaluation metrics
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier

REPO_ROOT = Path(__file__).resolve().parents[3]


class BioNexusModelAdapter:
    """Production Model Adapter wrapping the MOSAIC / BioNexus classifier."""

    SUBTYPES = PAM50_SUBTYPES

    def __init__(
        self,
        model: Any,
        model_uri: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model
        self.model_uri = model_uri
        self.metadata = metadata or {}
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def load(
        cls,
        model_uri: str = "models:/mosaic-subtype-classifier/Production",
        tracking_uri: str | None = None,
    ) -> BioNexusModelAdapter:
        """Loads champion model from MLflow registry or local fallback with cross-platform path resolution."""
        import mlflow
        import mlflow.pytorch
        from mlflow.client import MlflowClient

        if not tracking_uri:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            db_path = REPO_ROOT / "mlflow.db"
            tracking_uri = f"sqlite:///{db_path.resolve()}" if db_path.exists() else "sqlite:///:memory:"

        mlflow.set_tracking_uri(tracking_uri)

        # Retrieve model version metadata
        meta: dict[str, Any] = {
            "model_uri": model_uri,
            "tracking_uri": tracking_uri,
            "subtypes": cls.SUBTYPES,
        }

        loaded_model = None
        try:
            loaded_model = mlflow.pytorch.load_model(model_uri)
        except Exception as exc:
            # Cross-platform resolution (Windows host vs Docker container /mlruns)
            try:
                client = MlflowClient(tracking_uri=tracking_uri)
                parts = model_uri.replace("models:/", "").split("/")
                m_name = parts[0]
                stage_or_ver = parts[1] if len(parts) > 1 else "Production"
                mvs = client.get_latest_versions(m_name, stages=[stage_or_ver])
                if mvs:
                    mv = mvs[0]
                    meta["version"] = mv.version
                    meta["current_stage"] = mv.current_stage
                    meta["tags"] = mv.tags
                    raw_src = mv.source.replace("file:", "").replace("\\", "/")
                    if "mlruns/" in raw_src:
                        suffix = raw_src[raw_src.index("mlruns/") + len("mlruns/") :]
                        candidates = [
                            Path("/mlruns") / suffix,
                            REPO_ROOT / "mlruns" / suffix,
                        ]
                        for cand in candidates:
                            if cand.exists() and (cand / "MLmodel").exists():
                                loaded_model = mlflow.pytorch.load_model(str(cand))
                                break
            except Exception:
                pass

            if loaded_model is None:
                raise RuntimeError(f"Failed to load model from {model_uri}: {exc}")

        # Extract MLflow run metadata if available
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            parts = model_uri.replace("models:/", "").split("/")
            m_name = parts[0]
            stage_or_ver = parts[1] if len(parts) > 1 else "Production"
            mvs = client.get_latest_versions(m_name, stages=[stage_or_ver])
            if mvs:
                mv = mvs[0]
                meta["version"] = mv.version
                meta["current_stage"] = mv.current_stage
                meta["tags"] = mv.tags
                if mv.run_id:
                    run = client.get_run(mv.run_id)
                    meta["params"] = run.data.params
                    meta["metrics"] = run.data.metrics
        except Exception:
            pass

        return cls(model=loaded_model, model_uri=model_uri, metadata=meta)

    def predict_proba(
        self,
        g_rna: Any,
        g_mirna: Any,
        g_meth: Any,
    ) -> np.ndarray:
        """Executes forward pass and returns softmax subtype probabilities.

        Args:
            g_rna: PyG Data or batch for RNA layer
            g_mirna: PyG Data or batch for miRNA layer
            g_meth: PyG Data or batch for DNA methylation layer

        Returns:
            np.ndarray of shape (N, 5) with class probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            logits, _, _, _ = self.model(g_rna, g_mirna, g_meth)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(
        self,
        g_rna: Any,
        g_mirna: Any,
        g_meth: Any,
    ) -> list[str]:
        """Returns the highest-probability PAM50 subtype string for each input sample."""
        probs = self.predict_proba(g_rna, g_mirna, g_meth)
        indices = np.argmax(probs, axis=-1)
        return [self.SUBTYPES[int(idx)] for idx in indices]

    def get_metadata(self) -> dict[str, Any]:
        """Returns model contract metadata including version, tags, and expected dimensions."""
        return {
            "model_uri": self.model_uri,
            "subtypes": self.SUBTYPES,
            "expected_dimensions": {
                "rna": 5000,
                "mirna": 1881,
                "meth": 3957,
            },
            **self.metadata,
        }
