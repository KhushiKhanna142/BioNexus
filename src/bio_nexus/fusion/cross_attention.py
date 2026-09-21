"""
bio_nexus.fusion.cross_attention
-------------------------------------
Fuses the 3 per-layer patient embeddings (RNA, methylation, miRNA - each
[num_patients, embed_dim], from encoding/graph_transformer.py) into a
single fused per-patient representation, using learned cross-modal
attention: each modality's embedding attends to the other two, so the
model learns per-patient how much to weight each omics layer.

This also produces the modality-attention weights used for
interpretability (e.g. "for this patient, RNA mattered most") - these
weights are what the frontend's modality-contribution chart will read.

Design: treats the 3 modality embeddings for one patient as a length-3
"sequence" and runs standard multi-head self-attention over them. This
is a natural, minimal way to let each modality attend to the others
without hand-designing a fusion rule.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionConfig:
    embed_dim: int = 64       # must match GraphTransformerConfig.embed_dim
    num_heads: int = 4
    dropout: float = 0.3
    num_modalities: int = 3   # RNA, methylation, miRNA


class CrossAttentionFusion(nn.Module):
    """Fuses per-modality patient embeddings via multi-head self-attention
    across modalities. Input: 3 tensors of shape [num_patients, embed_dim].
    Output: fused embedding [num_patients, embed_dim] + attention weights
    [num_patients, num_modalities, num_modalities] for interpretability."""

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable per-modality identity embeddings, added to each modality's
        # vector before attention - lets the model distinguish "this is the
        # RNA embedding" from "this is the methylation embedding", since
        # otherwise attention has no way to tell the 3 inputs apart.
        self.modality_embedding = nn.Parameter(torch.randn(cfg.num_modalities, cfg.embed_dim) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
        )
        self.dropout = nn.Dropout(cfg.dropout)

        # Pools the 3 attended modality tokens into ONE fused vector per patient.
        # Learned weighted pooling (not a simple mean) so the model can also
        # learn a global preference across modalities, on top of the
        # per-patient attention weights.
        self.pool_weights = nn.Linear(cfg.embed_dim, 1)

    def forward(self, rna_emb: torch.Tensor, mirna_emb: torch.Tensor, meth_emb: torch.Tensor):
        """
        Each input: [num_patients, embed_dim]
        Returns:
            fused: [num_patients, embed_dim]
            modality_attn: [num_patients, num_heads, num_modalities, num_modalities]
                           - attention weights, useful for interpretability
                           (averaged over heads before returning)
        """
        num_patients = rna_emb.shape[0]

        # Stack into a length-3 "sequence" per patient: [num_patients, 3, embed_dim]
        modality_seq = torch.stack([rna_emb, mirna_emb, meth_emb], dim=1)
        modality_seq = modality_seq + self.modality_embedding.unsqueeze(0)  # add modality identity

        attn_out, attn_weights = self.attention(
            modality_seq, modality_seq, modality_seq,
            average_attn_weights=True,  # average over heads -> [num_patients, 3, 3]
        )
        h = self.norm1(modality_seq + self.dropout(attn_out))
        h = self.norm2(h + self.dropout(self.ffn(h)))  # [num_patients, 3, embed_dim]

        # Learned weighted pooling across the 3 modality tokens -> single fused vector
        pool_logits = self.pool_weights(h).squeeze(-1)          # [num_patients, 3]
        pool_weights = torch.softmax(pool_logits, dim=-1)       # [num_patients, 3]
        fused = torch.sum(h * pool_weights.unsqueeze(-1), dim=1)  # [num_patients, embed_dim]

        return fused, attn_weights, pool_weights


def get_modality_contribution(pool_weights: torch.Tensor) -> dict:
    """Convenience function for the frontend: converts the per-patient
    pooling weights into a readable {modality: weight} dict per patient.
    pool_weights: [num_patients, 3] (order: RNA, miRNA, methylation)"""
    modality_names = ["rna", "mirna", "meth"]
    weights = pool_weights.detach().cpu().numpy()
    return [
        {modality_names[i]: float(weights[p, i]) for i in range(3)}
        for p in range(weights.shape[0])
    ]


if __name__ == "__main__":
    # Smoke test with random embeddings (same shape the untrained
    # GraphGPS encoders currently produce) - confirms fusion module wiring.
    cfg = FusionConfig()
    fusion = CrossAttentionFusion(cfg)

    num_patients = 646
    rna_emb = torch.randn(num_patients, cfg.embed_dim)
    mirna_emb = torch.randn(num_patients, cfg.embed_dim)
    meth_emb = torch.randn(num_patients, cfg.embed_dim)

    fused, attn_weights, pool_weights = fusion(rna_emb, mirna_emb, meth_emb)

    print(f"Fused embedding shape: {tuple(fused.shape)} (patients x embed_dim)")
    print(f"Attention weights shape: {tuple(attn_weights.shape)} (patients x 3 modalities x 3 modalities)")
    print(f"Pooling weights shape: {tuple(pool_weights.shape)} (patients x 3 modalities)")
    print(f"\nExample - modality contribution for patient 0: {get_modality_contribution(pool_weights)[0]}")
    print("\nNOTE: fusion module is UNTRAINED at this point - weights are random-initialized, "
          "not meaningful until trained end-to-end with the classifier (see models/classifier.py + train.py)")