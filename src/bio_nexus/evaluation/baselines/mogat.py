"""
bio_nexus.evaluation.baselines.mogat
------------------------------------
MOGAT (Tanvir et al., Int. J. Mol. Sci. 2024, 25(5):2788), re-implemented from the
paper and the official repo (github.com/MezbahJUCSE39/MOGAT: mogat1.py, lib/module2.py).

How MOGAT actually works (this differs from how your writeup table describes it):

  Stage 1 - one GAT per omics graph. Each GAT is 2 layers, single head
            (GATConv(in, 512) -> ReLU -> dropout(0.5) -> GATConv(512, n_classes)).
            Node features are the CONCATENATION of ALL omics matrices, so cross-omics
            information enters at the feature level; only the graph differs per omics.
            The 512-d output of the FIRST layer is kept as that graph's embedding.
            Trained with Adam, CE loss, max 500 epochs, min 200, patience 30 on
            validation loss; the embedding is taken at the best-validation-loss epoch.
  Stage 2 - the per-graph embeddings are CONCATENATED (plus the raw features,
            addRawFeat=True in the official script) and classified by an MLP whose
            hyper-parameters come from a randomised search (macro-F1).

  There is NO VCDN in MOGAT. Graphs are patient-similarity graphs built from the top-3
  most similar patients by Pearson correlation.

Deviations (state them in the paper):
  * only 3 omics layers (rna, mirna, meth) instead of the paper's 8 data types;
  * GAT learning-rate grid reduced to one value by default (set mogat_lrs to the
    paper's {1e-2, 1e-3, 1e-4} for the full grid - roughly 3x slower);
  * MLP random search reduced (n_iter / cv set by MogatConfig), and grid values of
    the official search that are plainly unusable (learning rates 1, 2, 3) dropped;
  * edge weights are not used (the official call passes them to GATConv, which has
    no edge_dim and therefore ignores them);
  * graph edges symmetrised (assumption - the official edge files are pre-built);
  * early stopping uses task.stop_idx. NOTE: as I read mogat1.py, the official
    final embedding pass sets valid_mask = test_idx, i.e. the saved embedding is
    chosen by loss on the TEST nodes. Verify this yourself before saying it in the
    paper; either way it is another reason to re-run MOGAT under your protocol
    rather than quote its published numbers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

from .common import DataBundle, FoldTask, get_device


@dataclass
class MogatConfig:
    graph: str = "paper"  # "paper": top-k Pearson graphs | "bionexus": your k=10 cosine graphs
    top_k: int = 3
    hidden: int = 512
    lrs: tuple = (1e-3,)
    max_epochs: int = 500
    min_epochs: int = 200
    patience: int = 30
    add_raw: bool = True
    search_iter: int = 8
    search_cv: int = 3


def pearson_topk_edges(x: np.ndarray, k: int) -> torch.Tensor:
    """Top-k most Pearson-correlated patients per patient -> undirected edge_index."""
    xc = x - x.mean(axis=1, keepdims=True)
    xn = xc / (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-12)
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    nbr = np.argpartition(-sim, k, axis=1)[:, :k]
    src = nbr.reshape(-1)
    dst = np.repeat(np.arange(len(x)), k)
    return to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long))


class _GATNet(torch.nn.Module):
    def __init__(self, in_size: int, hid: int, out: int):
        super().__init__()
        self.conv1 = GATConv(in_size, hid, heads=1, concat=False)
        self.conv2 = GATConv(hid, out, heads=1, concat=False)

    def forward(self, x, edge_index):
        emb = self.conv1(x, edge_index)
        h = F.dropout(F.relu(emb), p=0.5, training=self.training)
        return self.conv2(h, edge_index), emb


def _train_gat_embedding(x, edge_index, y_t, fit_t, stop_t, cfg: MogatConfig, lr: float, n_classes: int):
    model = _GATNet(x.shape[1], cfg.hidden, n_classes).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_emb, bad = float("inf"), None, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        out, _ = model(x, edge_index)
        F.cross_entropy(out[fit_t], y_t[fit_t]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out, emb = model(x, edge_index)
            vloss = F.cross_entropy(out[stop_t], y_t[stop_t]).item()
        if vloss < best_loss:
            best_loss, best_emb, bad = vloss, emb.detach().cpu().numpy(), 0
        else:
            bad += 1
        if epoch >= cfg.min_epochs and bad >= cfg.patience:
            break
    return best_loss, best_emb


def fit_predict_mogat(data: DataBundle, task: FoldTask, cfg: MogatConfig, verbose: bool = False) -> dict:
    device = get_device()
    y = data.y
    n_classes = int(y.max()) + 1
    layers = data.layer_arrays()  # rna, mirna, meth
    x_all = torch.tensor(np.hstack(layers), dtype=torch.float32, device=device)  # concatenated node features

    if cfg.graph == "paper":
        edge_indices = [pearson_topk_edges(a, cfg.top_k) for a in layers]
    elif cfg.graph == "bionexus":
        edge_indices = [g.edge_index.cpu() for g in data.graphs]
    else:
        raise ValueError(cfg.graph)

    y_t = torch.tensor(y, dtype=torch.long, device=device)
    fit_t = torch.as_tensor(task.fit_idx, device=device)
    stop_t = torch.as_tensor(task.stop_idx, device=device)

    embs, chosen_lrs = [], []
    for gi, ei in enumerate(edge_indices):
        ei = ei.to(device)
        best = (float("inf"), None, None)
        for lr in cfg.lrs:  # pick lr by best validation loss (as in the official script)
            loss, emb = _train_gat_embedding(x_all, ei, y_t, fit_t, stop_t, cfg, lr, n_classes)
            if emb is not None and loss < best[0]:
                best = (loss, emb, lr)
        if best[1] is None:
            raise RuntimeError(f"GAT {gi} produced no valid embedding (NaN loss?)")
        embs.append(best[1])
        chosen_lrs.append(best[2])
        if verbose:
            print(f"      [mogat] graph {gi}: val_loss={best[0]:.4f} lr={best[2]}")

    z = np.hstack(embs + ([np.hstack(layers)] if cfg.add_raw else []))

    seed = task.seed * 1000 + task.fold
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(solver="adam", activation="relu", early_stopping=True, random_state=seed)),
        ]
    )
    # subset of the official randomised-search grid (sensible values only)
    space = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (256,), (512,), (128, 32), (256, 32)],
        "mlp__learning_rate_init": [1e-2, 1e-3, 1e-4],
        "mlp__max_iter": [250, 500, 1000],
        "mlp__n_iter_no_change": [10, 20, 50],
    }
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=cfg.search_iter,
        scoring="f1_macro",
        cv=StratifiedKFold(cfg.search_cv, shuffle=True, random_state=seed),
        random_state=seed,
        error_score=0.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(z[task.fit_idx], y[task.fit_idx])
    pred = search.predict(z[task.test_idx])
    return {
        "pred": pred,
        "mogat_graph": cfg.graph,
        "gat_lrs": chosen_lrs,
        "mlp_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
    }
