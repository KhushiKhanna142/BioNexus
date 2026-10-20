"""
bio_nexus.evaluation.baselines.pyg_models
-----------------------------------------
Controlled encoder swaps for the fused BioNexus model.

Everything except the per-layer graph encoder is IDENTICAL to BioNexusClassifier:
same input projection + LayerNorm, same depth/width/dropout, same output
projection + LayerNorm, same cross-attention fusion, same classifier head, same
graphs, same optimiser and training loop.

  graphgps : your model (GINE local message passing + GLOBAL multi-head attention)
  gat      : GATConv blocks    -> local attention only  (the MOGAT-style operator)
  gcn      : GCNConv blocks    -> fixed local averaging (the MOGONET-style operator)
  gine     : GINEConv blocks   -> your SAME local operator, but with the global
                                  attention REMOVED. This is the cleanest test of
                                  "does global attention help?", because it changes
                                  one thing only. (graphgps vs gat changes two:
                                  the local operator AND the global attention.)

Parameter counts differ (GPSConv layers also contain attention + an FFN). They are
printed at runtime; report them in the paper so capacity is not a hidden confound.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.nn import GATConv, GCNConv, GINEConv

from bio_nexus.models.classifier import BioNexusClassifier
from bio_nexus.train import TrainConfig

from .common import DataBundle, FoldTask, balanced_class_weights, get_device

ENCODER_KINDS = ("graphgps", "gat", "gcn", "gine")


def _make_local_conv(kind: str, cfg) -> nn.Module:
    h = cfg.hidden_dim
    if kind == "gat":
        # heads concatenated -> output width = hidden_dim; edge_dim=1 uses the similarity weight
        return GATConv(h, h // cfg.num_heads, heads=cfg.num_heads, concat=True, edge_dim=1, dropout=cfg.dropout)
    if kind == "gcn":
        return GCNConv(h, h)
    if kind == "gine":
        mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        return GINEConv(mlp, edge_dim=1)
    raise ValueError(kind)


class LocalOnlyEncoder(nn.Module):
    """Drop-in replacement for GraphGPSEncoder: same interface and same wrapper
    (input_proj -> LayerNorm -> ReLU -> dropout ... output_proj -> LayerNorm),
    but every layer is a purely local message-passing block with a residual
    connection and LayerNorm, and NO global attention."""

    def __init__(self, in_features: int, cfg, kind: str):
        super().__init__()
        self.kind = kind
        h = cfg.hidden_dim
        self.input_proj = nn.Linear(in_features, h)
        self.input_norm = nn.LayerNorm(h)
        self.convs = nn.ModuleList([_make_local_conv(kind, cfg) for _ in range(cfg.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(cfg.num_layers)])
        self.output_proj = nn.Linear(h, cfg.embed_dim)
        self.output_norm = nn.LayerNorm(cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, edge_index, edge_weight):
        h = self.dropout(F.relu(self.input_norm(self.input_proj(x))))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        edge_attr = edge_weight.unsqueeze(-1)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            if self.kind == "gcn":
                out = F.relu(conv(h, edge_index, edge_weight))
            elif self.kind == "gat":
                out = F.elu(conv(h, edge_index, edge_attr))
            else:  # gine: its MLP already contains the non-linearity
                out = conv(h, edge_index, edge_attr)
            h = norm(h + self.dropout(out))
        return self.output_norm(self.output_proj(h))


def build_fused_model(kind: str, rna_dim: int, mirna_dim: int, meth_dim: int) -> BioNexusClassifier:
    model = BioNexusClassifier(rna_dim, mirna_dim, meth_dim)
    if kind != "graphgps":
        enc_cfg = model.cfg.encoder_cfg
        model.rna_encoder = LocalOnlyEncoder(rna_dim, enc_cfg, kind)
        model.mirna_encoder = LocalOnlyEncoder(mirna_dim, enc_cfg, kind)
        model.meth_encoder = LocalOnlyEncoder(meth_dim, enc_cfg, kind)
    return model


def fit_predict_fused(
    kind: str, data: DataBundle, task: FoldTask, cfg: TrainConfig, verbose: bool = False
) -> dict:
    """Train one fused model on task.fit_idx, early-stop on task.stop_idx
    (macro-F1, patience from cfg), return predictions on task.test_idx.
    Mirrors train.py: full-batch, transductive, AdamW, class-weighted CE."""
    device = get_device()
    graphs = [g.to(device) for g in data.graphs]
    model = build_fused_model(kind, *[g.x.shape[1] for g in graphs]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    y = data.y
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    fit_t = torch.as_tensor(task.fit_idx, device=device)
    stop_t = torch.as_tensor(task.stop_idx, device=device)
    test_t = torch.as_tensor(task.test_idx, device=device)

    class_w = torch.tensor(balanced_class_weights(y[task.fit_idx]), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        logits, _, _, _ = model(*graphs)
        loss = loss_fn(logits[fit_t], y_t[fit_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(*graphs)
            stop_pred = logits[stop_t].argmax(-1).cpu().numpy()
        f1 = f1_score(y[task.stop_idx], stop_pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (epoch % 25 == 0):
            print(f"      [{kind}] epoch {epoch:3d} loss={loss.item():.4f} stop_f1={f1:.4f}")
        if bad >= cfg.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _, _, pool_w = model(*graphs)
        pred = logits[test_t].argmax(-1).cpu().numpy()
        pool_mean = pool_w[test_t].mean(0).cpu().numpy()  # mean modality weight on test nodes

    return {
        "pred": pred,
        "best_epoch": int(best_epoch),
        "best_stop_f1": float(best_f1),
        "n_params": int(n_params),
        "mean_pool_weight_rna_mirna_meth": [float(v) for v in pool_mean],
    }
