"""
bio_nexus.evaluation.baselines.mogonet
--------------------------------------
Port of MOGONET (Wang et al., Nat. Commun. 2021) following the official code
(github.com/txWang/MOGONET: models.py, train_test.py, utils.py) line by line:

  * per-view graph from cosine distance, thresholded so that on average ~10 edges
    per node survive (the threshold is computed on the TRAINING nodes only);
  * test-time graph over train+test nodes containing ONLY train<->test edges
    (no train-train and no test-test edges - this is what the official code does);
  * 3-layer GCN encoder [400, 400, 200] (BRCA setting), LeakyReLU(0.25), dropout 0.5;
  * one linear classifier per view + VCDN fusion (cross-view outer product over the
    sigmoid'd per-view class scores -> hidden layer of size 5^3 -> 5 classes);
  * 500 epochs of per-view pre-training, then 2500 epochs of joint training with
    lr_e=5e-4, lr_c=1e-3, evaluated at the FINAL epoch (the official recipe has
    no validation set and no early stopping).

Deliberate differences from the official repo (state them in the paper):
  * input features are YOUR preprocessed matrices (5000 variance-filtered RNA genes,
    1881 miRNAs, 3957 methylation probes), not MOGONET's own feature-selected files;
  * dense adjacency instead of torch.sparse (identical maths, N is only ~650);
  * `np.asscalar` (removed from NumPy) replaced by `.item()`.

SAMPLE WEIGHTS: the official code weights each training sample by its class
FREQUENCY (count/total), which up-weights the majority class. `weighting="official"`
keeps that. `weighting="balanced"` uses inverse-frequency weights instead, which is
the fairer setting for macro-F1 on PAM50. Report official as the main row and
balanced as a sensitivity row if the two differ a lot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import N_CLASSES, DataBundle, FoldTask, balanced_class_weights, get_device


@dataclass
class MogonetConfig:
    adj_parameter: int = 10  # edges per node (BRCA setting in the official repo)
    dim_he_list: tuple = (400, 400, 200)  # BRCA setting
    dropout: float = 0.5
    pretrain_epochs: int = 500
    epochs: int = 2500
    lr_e_pretrain: float = 1e-3
    lr_e: float = 5e-4
    lr_c: float = 1e-3
    weighting: str = "official"  # "official" | "balanced"


# ----------------------------------------------------------------------------
# graph construction (utils.py)
# ----------------------------------------------------------------------------
def _cosine_distance(x1: torch.Tensor, x2: torch.Tensor | None = None, eps: float = 1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def _adj_threshold(edge_per_node: int, data: torch.Tensor) -> float:
    dist = _cosine_distance(data, data)
    return torch.sort(dist.reshape(-1)).values[edge_per_node * data.shape[0]].item()


def _symmetrise_and_normalise(adj: torch.Tensor) -> torch.Tensor:
    adj_t = adj.transpose(0, 1)
    eye = torch.eye(adj.shape[0], device=adj.device)
    adj = adj + adj_t * (adj_t > adj).float() - adj * (adj_t > adj).float()
    return F.normalize(adj + eye, p=1)  # row-wise L1 normalisation, self-loops added


def _adj_train(data: torch.Tensor, parameter: float) -> torch.Tensor:
    dist = _cosine_distance(data, data)
    g = (dist <= parameter).float()
    g.fill_diagonal_(0)
    return _symmetrise_and_normalise((1 - dist) * g)


def _adj_test(data_trte: torch.Tensor, num_tr: int, parameter: float) -> torch.Tensor:
    n = data_trte.shape[0]
    adj = torch.zeros((n, n), device=data_trte.device)
    d_tr2te = _cosine_distance(data_trte[:num_tr], data_trte[num_tr:])
    adj[:num_tr, num_tr:] = (1 - d_tr2te) * (d_tr2te <= parameter).float()
    d_te2tr = _cosine_distance(data_trte[num_tr:], data_trte[:num_tr])
    adj[num_tr:, :num_tr] = (1 - d_te2tr) * (d_te2tr <= parameter).float()
    return _symmetrise_and_normalise(adj)


def _official_sample_weight(labels: np.ndarray) -> np.ndarray:
    count = np.array([np.sum(labels == c) for c in range(N_CLASSES)], dtype=np.float64)
    w = np.zeros(labels.shape, dtype=np.float64)
    for c in range(N_CLASSES):
        w[labels == c] = count[c] / count.sum()
    return w


# ----------------------------------------------------------------------------
# model (models.py)
# ----------------------------------------------------------------------------
def _xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class _GraphConv(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(i, o))
        self.bias = nn.Parameter(torch.zeros(o))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x, adj):
        return torch.mm(adj, torch.mm(x, self.weight)) + self.bias


class _GCNE(nn.Module):
    def __init__(self, in_dim: int, dims, dropout: float):
        super().__init__()
        self.gc1 = _GraphConv(in_dim, dims[0])
        self.gc2 = _GraphConv(dims[0], dims[1])
        self.gc3 = _GraphConv(dims[1], dims[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj), 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc2(x, adj), 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        return F.leaky_relu(self.gc3(x, adj), 0.25)


class _VCDN(nn.Module):
    def __init__(self, num_view: int, num_cls: int, hidden: int):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(num_cls**num_view, hidden), nn.LeakyReLU(0.25), nn.Linear(hidden, num_cls)
        )
        self.model.apply(_xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        in_list = [torch.sigmoid(t) for t in in_list]
        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)), (-1, self.num_cls**2, 1)
        )
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, self.num_cls ** (i + 1), 1))
        return self.model(torch.reshape(x, (-1, self.num_cls**num_view)))


def _make_models(dims_in, cfg: MogonetConfig, device):
    num_view = len(dims_in)
    enc = [_GCNE(d, cfg.dim_he_list, cfg.dropout).to(device) for d in dims_in]
    clf = []
    for _ in range(num_view):
        lin = nn.Linear(cfg.dim_he_list[-1], N_CLASSES)
        _xavier_init(lin)
        clf.append(lin.to(device))
    vcdn = _VCDN(num_view, N_CLASSES, N_CLASSES**num_view).to(device)
    return enc, clf, vcdn


def _make_optims(enc, clf, vcdn, lr_e: float, lr_c: float):
    opt_views = [
        torch.optim.Adam(list(e.parameters()) + list(c.parameters()), lr=lr_e) for e, c in zip(enc, clf, strict=True)
    ]
    return opt_views, torch.optim.Adam(vcdn.parameters(), lr=lr_c)


def _train_epoch(xs, adjs, label, sw, enc, clf, vcdn, opt_views, opt_vcdn, train_vcdn: bool):
    crit = nn.CrossEntropyLoss(reduction="none")
    for m in [*enc, *clf, vcdn]:
        m.train()
    for i in range(len(xs)):  # per-view losses (this is what trains encoders + view classifiers)
        opt_views[i].zero_grad()
        loss = torch.mean(crit(clf[i](enc[i](xs[i], adjs[i])), label) * sw)
        loss.backward()
        opt_views[i].step()
    if train_vcdn:  # fusion loss: only the VCDN parameters are stepped
        opt_vcdn.zero_grad()
        out = vcdn([clf[i](enc[i](xs[i], adjs[i])) for i in range(len(xs))])
        loss = torch.mean(crit(out, label) * sw)
        loss.backward()
        opt_vcdn.step()


def fit_predict_mogonet(data: DataBundle, task: FoldTask, cfg: MogonetConfig, verbose: bool = False) -> dict:
    device = get_device()
    y = data.y
    tr, te = task.fit_idx, task.test_idx
    num_tr = len(tr)
    order = np.concatenate([tr, te])  # official layout: [train nodes ..., test nodes ...]

    layers = [torch.tensor(a, dtype=torch.float32, device=device) for a in data.layer_arrays()]
    tr_t = torch.as_tensor(tr, device=device)
    order_t = torch.as_tensor(order, device=device)
    x_tr = [x[tr_t] for x in layers]
    x_trte = [x[order_t] for x in layers]

    adj_tr, adj_te = [], []
    for xt, xtt in zip(x_tr, x_trte, strict=True):
        thr = _adj_threshold(cfg.adj_parameter, xt)  # threshold from TRAIN nodes only
        adj_tr.append(_adj_train(xt, thr))
        adj_te.append(_adj_test(xtt, num_tr, thr))

    label = torch.tensor(y[tr], dtype=torch.long, device=device)
    if cfg.weighting == "official":
        sw_np = _official_sample_weight(y[tr])
    elif cfg.weighting == "balanced":
        sw_np = balanced_class_weights(y[tr])[y[tr]]
    else:
        raise ValueError(cfg.weighting)
    sw = torch.tensor(sw_np, dtype=torch.float32, device=device)

    enc, clf, vcdn = _make_models([x.shape[1] for x in x_tr], cfg, device)

    opt_views, opt_vcdn = _make_optims(enc, clf, vcdn, cfg.lr_e_pretrain, cfg.lr_c)
    for _ in range(cfg.pretrain_epochs):
        _train_epoch(x_tr, adj_tr, label, sw, enc, clf, vcdn, opt_views, opt_vcdn, train_vcdn=False)

    opt_views, opt_vcdn = _make_optims(enc, clf, vcdn, cfg.lr_e, cfg.lr_c)
    for epoch in range(cfg.epochs + 1):
        _train_epoch(x_tr, adj_tr, label, sw, enc, clf, vcdn, opt_views, opt_vcdn, train_vcdn=True)
        if verbose and epoch % 500 == 0:
            print(f"      [mogonet] epoch {epoch}")

    for m in [*enc, *clf, vcdn]:
        m.eval()
    with torch.no_grad():
        out = vcdn([clf[i](enc[i](x_trte[i], adj_te[i])) for i in range(len(x_trte))])
        pred = out[num_tr:].argmax(1).cpu().numpy()  # rows num_tr: are the test nodes, in test_idx order
    return {"pred": pred, "weighting": cfg.weighting}
