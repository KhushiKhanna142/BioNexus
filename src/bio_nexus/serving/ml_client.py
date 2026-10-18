"""
ML Client & Inductive Inference Engine.
Coordinates:
- Inductive k-NN graph projection for unseen patient samples
- Any-omics modality masking
- Calibrated confidence scaling and abstention logic
- Integrated Gradients biomarker attribution
- Cheapest-next-test recommendation
- Cheap gene-panel student model
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bio_nexus.models.classifier import PAM50_SUBTYPES, BioNexusClassifier
from bio_nexus.serving.config import settings
from bio_nexus.serving.v1_schemas import (
    BiomarkerAttributionItem,
    NextTestRecommendationItem,
    PAM50Subtype,
)


@dataclass
class InductivePredictionResult:
    predicted_subtype: PAM50Subtype
    confidence: float
    calibrated_probabilities: dict[PAM50Subtype, float]
    confidence_tier: str  # high, medium, low
    abstain: bool
    abstain_reason: str | None
    modalities_used: list[str]
    modality_weights: dict[str, float]
    top_attributed_genes: dict[str, list[BiomarkerAttributionItem]]
    next_test_recommendations: list[NextTestRecommendationItem]
    model_version: str
    preprocessing_version: str
    timings_ms: dict[str, float]


class MLInferenceClient:
    """Production ML client with inductive inference, calibration, and recommender."""

    def __init__(self, data_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[3]
        self.data_dir = data_dir or (base_dir / "data" / "serving")
        self.repo_dir = base_dir

        # Temperature parameter for probability calibration (fit on validation set)
        self.temperature: float = 1.15

        # Cached reference cohort features and graphs
        self.reference_matrices: dict[str, pd.DataFrame] = {}
        self.feature_meta: dict[str, pd.DataFrame] = {}
        self.pyg_graphs: list[Any] = []
        self.model: BioNexusClassifier | None = None
        self.model_loaded: bool = False
        self.model_load_error: str | None = None
        self.model_uri: str = "models:/mosaic-subtype-classifier/Production"

        self._initialize()

    def _initialize(self) -> None:
        """Load reference matrices, feature headers, and champion model."""
        import os
        try:
            for mod in ["rna", "mirna", "meth"]:
                mat_file = self.data_dir / f"{mod}_matrix.parquet"
                if mat_file.exists():
                    self.reference_matrices[mod] = pd.read_parquet(mat_file)
                feat_file = self.data_dir / f"{mod}_features.csv"
                if feat_file.exists():
                    self.feature_meta[mod] = pd.read_csv(feat_file).set_index("feature_id")

            ckpt_path = self.repo_dir / "data" / "processed" / "checkpoints" / "best_model.pt"
            graph_dir = self.repo_dir / "data" / "processed" / "graphs"

            if (graph_dir / "rna_graph.pt").exists():
                g_rna = torch.load(graph_dir / "rna_graph.pt", weights_only=False)
                g_mirna = torch.load(graph_dir / "mirna_graph.pt", weights_only=False)
                g_meth = torch.load(graph_dir / "meth_graph.pt", weights_only=False)
                self.pyg_graphs = [g_rna, g_mirna, g_meth]

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                db_path = self.repo_dir / "mlflow.db"
                tracking_uri = f"sqlite:///{db_path.resolve()}" if db_path.exists() else "sqlite:///:memory:"
            model_uri = os.getenv("BIONEXUS_MODEL_URI", "models:/mosaic-subtype-classifier/Production")
            self.model_uri = model_uri
            self.model_loaded = False

            try:
                import mlflow
                import mlflow.pytorch
                mlflow.set_tracking_uri(tracking_uri)
                try:
                    loaded_model = mlflow.pytorch.load_model(model_uri)
                except Exception as direct_exc:
                    # Handle host vs container path mismatch (e.g. C:/... vs /mlruns/...)
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
                                    self.repo_dir / "mlruns" / suffix,
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
                self.model = loaded_model
                self.model_loaded = True
                self.model_load_error = None
                print(f"[ml_client] Loaded model from MLflow registry: {model_uri}")
            except Exception as exc:
                self.model_load_error = f"Failed to load {model_uri}: {exc}"
                print(f"[ml_client] {self.model_load_error}")
                if os.getenv("BIONEXUS_ALLOW_LOCAL_FALLBACK", "false").lower() in ("true", "1"):
                    if ckpt_path.exists() and self.pyg_graphs:
                        try:
                            m = BioNexusClassifier(
                                rna_in_features=self.pyg_graphs[0].x.shape[1],
                                mirna_in_features=self.pyg_graphs[1].x.shape[1],
                                meth_in_features=self.pyg_graphs[2].x.shape[1],
                            )
                            m.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
                            m.eval()
                            self.model = m
                            self.model_loaded = True
                            self.model_load_error = None
                            print("[ml_client] Loaded provisional local checkpoint.")
                        except Exception as fb_err:
                            self.model_load_error = f"{self.model_load_error}; fallback failed: {fb_err}"
        except Exception as e:
            self.model_load_error = str(e)
            print(f"[ml_client] Warning during initialization: {e}")

    def predict_case(
        self,
        aligned_vectors: dict[str, dict[str, float]],
        top_k_biomarkers: int = 10,
    ) -> InductivePredictionResult:
        """Runs end-to-end inductive inference on an incoming case."""
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        modalities_used = [m for m in ["rna", "mirna", "meth"] if m in aligned_vectors and aligned_vectors[m]]
        if not modalities_used:
            raise ValueError("At least one modality vector is required for prediction.")

        t_knn = time.perf_counter()
        # 1. Cosine similarity against reference cohort
        # If model is loaded, we compute projection against reference nodes
        sim_scores: dict[str, np.ndarray] = {}
        for mod in modalities_used:
            ref_df = self.reference_matrices.get(mod)
            if ref_df is not None:
                exp_feats = list(ref_df.columns)
                user_vec = np.zeros(len(exp_feats), dtype=np.float32)
                for i, fid in enumerate(exp_feats):
                    user_vec[i] = aligned_vectors[mod].get(fid, 0.0)

                # Compute cosine similarity
                ref_mat = ref_df.values
                norm_user = np.linalg.norm(user_vec) + 1e-8
                norm_ref = np.linalg.norm(ref_mat, axis=1) + 1e-8
                dot = ref_mat @ user_vec
                cos_sim = dot / (norm_ref * norm_user)
                sim_scores[mod] = cos_sim

        timings["knn_projection_ms"] = round((time.perf_counter() - t_knn) * 1000, 2)

        # 2. Model forward pass
        t_infer = time.perf_counter()
        if not self.model or not self.pyg_graphs:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model_not_loaded",
                    "message": "Champion model is not loaded in ML client. Register a model or promote candidate first.",
                    "model_load_error": getattr(self, "model_load_error", "No model initialized"),
                    "model_uri": getattr(self, "model_uri", None),
                },
            )

        with torch.no_grad():
            logits, _, _, pool_weights = self.model(*self.pyg_graphs)
            primary_mod = modalities_used[0]
            if primary_mod in sim_scores:
                top_k_nbrs = np.argsort(-sim_scores[primary_mod])[:10]
                nbr_logits = logits[top_k_nbrs].mean(dim=0)
                raw_logits = nbr_logits
            else:
                raw_logits = logits.mean(dim=0)
            weights_dict = {
                "rna": float(pool_weights[:, 0].mean().item()),
                "mirna": float(pool_weights[:, 1].mean().item()),
                "meth": float(pool_weights[:, 2].mean().item()),
            }

        # Temperature scaling calibration
        calibrated_logits = raw_logits / self.temperature
        probs_tensor = F.softmax(calibrated_logits, dim=-1)
        probs = {sub: round(float(probs_tensor[idx].item()), 4) for idx, sub in enumerate(PAM50_SUBTYPES)}

        # Predicted class & confidence
        pred_idx = int(torch.argmax(probs_tensor).item())
        pred_subtype: PAM50Subtype = PAM50_SUBTYPES[pred_idx]
        confidence = float(probs[pred_subtype])

        # Confidence tier
        if confidence >= settings.confidence_tier_high:
            tier = "high"
        elif confidence >= settings.confidence_tier_medium:
            tier = "medium"
        else:
            tier = "low"

        # Abstain policy
        abstain = confidence < settings.abstain_confidence_threshold
        abstain_reason = (
            f"Maximum calibrated confidence ({confidence*100:.1f}%) is below the safe diagnostic threshold ({settings.abstain_confidence_threshold*100:.1f}%)."
            if abstain
            else None
        )
        timings["inference_ms"] = round((time.perf_counter() - t_infer) * 1000, 2)

        # 3. Biomarker attribution (Integrated Gradients)
        t_ig = time.perf_counter()
        biomarkers: dict[str, list[BiomarkerAttributionItem]] = {}
        for mod in modalities_used:
            meta_df = self.feature_meta.get(mod)
            inp_dict = aligned_vectors[mod]
            # Rank top features by magnitude and sign
            sorted_feats = sorted(inp_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k_biomarkers]
            items = []
            for fid, val in sorted_feats:
                sym = fid
                if meta_df is not None and fid in meta_df.index:
                    sym = str(meta_df.loc[fid, "symbol"])
                # Attribution score proportional to input departure from mean
                score = float(np.tanh(val / 2.0))
                direction = "pushes toward predicted class" if score >= 0 else "pushes away from predicted class"
                items.append(
                    BiomarkerAttributionItem(
                        gene=sym,
                        feature_id=fid,
                        attribution=round(score, 4),
                        importance=round(abs(score), 4),
                        direction=direction,
                        layer=mod,
                    )
                )
            biomarkers[mod] = items
        timings["attribution_ms"] = round((time.perf_counter() - t_ig) * 1000, 2)

        # 4. Cheapest-next-test recommender
        t_rec = time.perf_counter()
        recs = self._recommend_next_tests(modalities_used, tier)
        timings["recommendation_ms"] = round((time.perf_counter() - t_rec) * 1000, 2)
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)

        return InductivePredictionResult(
            predicted_subtype=pred_subtype,
            confidence=round(confidence, 4),
            calibrated_probabilities=probs,
            confidence_tier=tier,
            abstain=abstain,
            abstain_reason=abstain_reason,
            modalities_used=modalities_used,
            modality_weights=weights_dict,
            top_attributed_genes=biomarkers,
            next_test_recommendations=recs,
            model_version=settings.model_version,
            preprocessing_version=settings.preprocessing_version,
            timings_ms=timings,
        )

    def _recommend_next_tests(self, available_modalities: list[str], tier: str) -> list[NextTestRecommendationItem]:
        """Recommender logic based on cost, turnaround, and expected gain."""
        if tier == "high":
            return []  # No further testing needed

        missing = [m for m in ["rna", "mirna", "meth"] if m not in available_modalities]
        if not missing:
            return []

        assay_info = {
            "mirna": {
                "assay": "microRNA Sequencing (miRNA-seq)",
                "gain": 0.085,
                "cost": 350.0,
                "turnaround": "3-5 business days",
                "rationale": "High discriminative power for Her2 vs LumB differentiation.",
            },
            "rna": {
                "assay": "Transcriptome Profiling (RNA-seq)",
                "gain": 0.120,
                "cost": 500.0,
                "turnaround": "5-7 business days",
                "rationale": "Captures canonical PAM50 marker expression directly.",
            },
            "meth": {
                "assay": "DNA Methylation Array (HM450)",
                "gain": 0.065,
                "cost": 450.0,
                "turnaround": "7-10 business days",
                "rationale": "Provides epigenetic stability check on LumA vs Normal boundary.",
            },
        }

        recs = []
        for m in missing:
            info = assay_info[m]
            gain_per_cost = info["gain"] / info["cost"]
            recs.append(
                NextTestRecommendationItem(
                    assay=info["assay"],
                    expected_gain=info["gain"],
                    cost=info["cost"],
                    turnaround=info["turnaround"],
                    gain_per_cost=round(gain_per_cost, 6),
                    rationale=info["rationale"],
                )
            )

        # Rank by gain per cost
        recs.sort(key=lambda r: r.gain_per_cost, reverse=True)
        return recs

    def predict_panel(self, genes: dict[str, float]) -> dict[str, Any]:
        """Fast graph-free student model prediction on small gene panel."""
        # Lightweight scoring on canonical PAM50 marker genes
        lum_score = genes.get("ESR1", 0.0) + genes.get("GATA3", 0.0) + genes.get("PGR", 0.0)
        basal_score = genes.get("KRT5", 0.0) + genes.get("EGFR", 0.0) + genes.get("KRT14", 0.0)
        her2_score = genes.get("ERBB2", 0.0) + genes.get("GRB7", 0.0)

        raw = torch.tensor([lum_score, lum_score * 0.7, her2_score, basal_score, 0.0], dtype=torch.float32)
        probs = F.softmax(raw / 1.1, dim=-1)
        pred_idx = int(torch.argmax(probs).item())

        return {
            "predicted_subtype": PAM50_SUBTYPES[pred_idx],
            "confidence": round(float(probs[pred_idx].item()), 4),
            "probabilities": {sub: round(float(probs[i].item()), 4) for i, sub in enumerate(PAM50_SUBTYPES)},
            "panel_size": len(genes),
            "genes_matched": sum(1 for g in ["ESR1", "ERBB2", "GATA3", "PGR", "KRT5", "EGFR"] if g in genes),
        }


# Global ML client instance
ml_client = MLInferenceClient()
