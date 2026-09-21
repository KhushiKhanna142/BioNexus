"""
bio_nexus.models.classifier
--------------------------------
End-to-end BioNexus model: wires together the 3 per-layer GraphGPS
encoders (encoding/graph_transformer.py), cross-attention fusion
(fusion/cross_attention.py), and a classification head predicting
PAM50 subtype (LumA, LumB, Her2, Basal, Normal).

This is a single nn.Module so the whole pipeline - graph encoding,
fusion, classification - trains END TO END with one loss/optimizer,
rather than each piece being trained separately. This matters: the
encoders and fusion module only become meaningful once trained
jointly against the actual classification task (see train.py).
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from bio_nexus.encoding.graph_transformer import GraphGPSEncoder, GraphTransformerConfig
from bio_nexus.fusion.cross_attention import CrossAttentionFusion, FusionConfig

PAM50_SUBTYPES = ["LumA", "LumB", "Her2", "Basal", "Normal"]


@dataclass
class ClassifierConfig:
    encoder_cfg: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
    fusion_cfg: FusionConfig = field(default_factory=FusionConfig)
    hidden_dim: int = 32          # classifier head hidden size
    dropout: float = 0.3
    num_classes: int = 5          # len(PAM50_SUBTYPES)


class BioNexusClassifier(nn.Module):
    """End-to-end multi-omics graph transformer classifier.

    forward() expects the 3 raw per-layer graphs (from
    graph/construction.py), runs each through its own GraphGPS encoder,
    fuses the 3 resulting embeddings via cross-attention, and predicts
    PAM50 subtype from the fused representation."""

    def __init__(self, rna_in_features: int, mirna_in_features: int, meth_in_features: int,
                 cfg: ClassifierConfig = None):
        super().__init__()
        self.cfg = cfg or ClassifierConfig()

        # One encoder per omics layer - separate weights, since each layer
        # has a different raw feature dimensionality and different biology.
        self.rna_encoder = GraphGPSEncoder(rna_in_features, self.cfg.encoder_cfg)
        self.mirna_encoder = GraphGPSEncoder(mirna_in_features, self.cfg.encoder_cfg)
        self.meth_encoder = GraphGPSEncoder(meth_in_features, self.cfg.encoder_cfg)

        self.fusion = CrossAttentionFusion(self.cfg.fusion_cfg)

        embed_dim = self.cfg.encoder_cfg.embed_dim
        self.classifier_head = nn.Sequential(
            nn.Linear(embed_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_classes),
        )

    def forward(self, rna_graph, mirna_graph, meth_graph):
        """Each *_graph is a torch_geometric.data.Data object with
        .x, .edge_index, .edge_attr (from graph/construction.py).
        Returns:
            logits: [num_patients, num_classes] - raw classification scores
            fused_embedding: [num_patients, embed_dim] - for downstream use (e.g. clustering, viz)
            attn_weights: [num_patients, 3, 3] - cross-modal attention, for interpretability
            pool_weights: [num_patients, 3] - per-modality contribution, for the frontend
        """
        rna_emb = self.rna_encoder(rna_graph.x, rna_graph.edge_index, rna_graph.edge_attr)
        mirna_emb = self.mirna_encoder(mirna_graph.x, mirna_graph.edge_index, mirna_graph.edge_attr)
        meth_emb = self.meth_encoder(meth_graph.x, meth_graph.edge_index, meth_graph.edge_attr)

        fused, attn_weights, pool_weights = self.fusion(rna_emb, mirna_emb, meth_emb)
        logits = self.classifier_head(fused)

        return logits, fused, attn_weights, pool_weights

    def predict_subtype(self, rna_graph, mirna_graph, meth_graph):
        """Convenience method for inference/serving: returns human-readable
        predicted subtype + confidence + per-modality contribution per patient."""
        self.eval()
        with torch.no_grad():
            logits, _, _, pool_weights = self.forward(rna_graph, mirna_graph, meth_graph)
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1)

        results = []
        for i in range(logits.shape[0]):
            results.append({
                "predicted_subtype": PAM50_SUBTYPES[pred_idx[i].item()],
                "confidence": float(probs[i, pred_idx[i]].item()),
                "all_probabilities": {
                    PAM50_SUBTYPES[c]: float(probs[i, c].item()) for c in range(len(PAM50_SUBTYPES))
                },
                "modality_contribution": {
                    "rna": float(pool_weights[i, 0].item()),
                    "mirna": float(pool_weights[i, 1].item()),
                    "meth": float(pool_weights[i, 2].item()),
                },
            })
        return results


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[3]
    GRAPH_DIR = BASE_DIR / "data" / "processed" / "graphs"

    rna_graph = torch.load(GRAPH_DIR / "rna_graph.pt", weights_only=False)
    mirna_graph = torch.load(GRAPH_DIR / "mirna_graph.pt", weights_only=False)
    meth_graph = torch.load(GRAPH_DIR / "meth_graph.pt", weights_only=False)

    model = BioNexusClassifier(
        rna_in_features=rna_graph.x.shape[1],
        mirna_in_features=mirna_graph.x.shape[1],
        meth_in_features=meth_graph.x.shape[1],
    )

    logits, fused, attn_weights, pool_weights = model(rna_graph, mirna_graph, meth_graph)
    print(f"Logits shape: {tuple(logits.shape)} (patients x {len(PAM50_SUBTYPES)} subtypes)")
    print(f"Fused embedding shape: {tuple(fused.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {total_params:,}")

    example = model.predict_subtype(rna_graph, mirna_graph, meth_graph)[0]
    print(f"\nExample prediction (patient 0, UNTRAINED - not meaningful yet):")
    print(f"  Predicted: {example['predicted_subtype']} ({example['confidence']:.1%} confidence)")
    print(f"  Modality contribution: {example['modality_contribution']}")
    print("\nNOTE: model is UNTRAINED - this confirms the full pipeline runs end-to-end "
          "with correct shapes. Real predictions require train.py next.")