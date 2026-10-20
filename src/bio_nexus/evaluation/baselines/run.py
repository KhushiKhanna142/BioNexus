"""
bio_nexus.evaluation.baselines.run
----------------------------------
Runs every method on IDENTICAL outer folds and writes one JSON per (method, seed).
Results are resumable: finished (method, seed) pairs are skipped unless --force.

Methods
  lr_concat            logistic regression on [rna|mirna|meth] concatenated (tuned C)
  rf_concat            random forest on the same
  lr_rna               logistic regression on RNA ONLY  <- label-circularity check:
                       PAM50 is defined from mRNA, so a linear model on RNA alone is the
                       baseline every reviewer will ask about
  mogonet              MOGONET (official recipe, see mogonet.py)
  mogat                MOGAT   (official recipe, see mogat.py)
  bionexus             your model: GraphGPS encoders + cross-attention fusion
  bionexus_gat         same model, GAT encoders       (local attention only)
  bionexus_gcn         same model, GCN encoders       (local averaging only)
  bionexus_gine_local  same model, GINE encoders with NO global attention
                       (isolates the effect of global attention)

Typical use (from the repo root):
  # smoke test, a few minutes:
  python -m bio_nexus.evaluation.baselines.run --methods lr_rna bionexus --seeds 42 --max-epochs 30
  # full, honest comparison (the numbers for the paper):
  python -m bio_nexus.evaluation.baselines.run --methods all --seeds 42 43 44 45 46 --protocol inner_val
  # reproduce train.py's number (optimistic protocol; do not tabulate as-is):
  python -m bio_nexus.evaluation.baselines.run --methods bionexus --seeds 42 --protocol eval_fold
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bio_nexus.train import TrainConfig, load_data

from .common import (
    DataBundle,
    FoldTask,
    compute_metrics,
    make_task,
    outer_folds,
    result_path,
    save_result,
    set_all_seeds,
)
from .mogat import MogatConfig, fit_predict_mogat
from .mogonet import MogonetConfig, fit_predict_mogonet
from .pyg_models import fit_predict_fused

# ----------------------------------------------------------------------------
# sklearn baselines (fit on fit_idx only; they do not use stop_idx)
# ----------------------------------------------------------------------------
def _fit_predict_sklearn(kind: str, data: DataBundle, task: FoldTask) -> dict:
    layers = data.layer_arrays()
    x = layers[0] if kind == "lr_rna" else np.hstack(layers)
    seed = task.seed * 1000 + task.fold
    if kind in ("lr_concat", "lr_rna"):
        clf = LogisticRegressionCV(
            Cs=[1e-3, 1e-2, 1e-1, 1.0],
            cv=3,
            scoring="f1_macro",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    elif kind == "rf_concat":
        clf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    else:
        raise ValueError(kind)
    model = make_pipeline(StandardScaler(), clf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x[task.fit_idx], data.y[task.fit_idx])
    return {"pred": model.predict(x[task.test_idx])}


# ----------------------------------------------------------------------------
# registry: name -> callable(data, task, args, train_cfg) -> dict with "pred"
# ----------------------------------------------------------------------------
def _build_registry(args):
    mogonet_cfg = MogonetConfig(
        pretrain_epochs=args.mogonet_pretrain_epochs,
        epochs=args.mogonet_epochs,
        weighting=args.mogonet_weighting,
    )
    mogat_cfg = MogatConfig(
        graph=args.mogat_graph,
        lrs=tuple(args.mogat_lrs),
        max_epochs=args.mogat_max_epochs,
        min_epochs=args.mogat_min_epochs,
        search_iter=args.mogat_search_iter,
    )
    v = args.verbose
    return {
        "lr_concat": lambda d, t, c: _fit_predict_sklearn("lr_concat", d, t),
        "rf_concat": lambda d, t, c: _fit_predict_sklearn("rf_concat", d, t),
        "lr_rna": lambda d, t, c: _fit_predict_sklearn("lr_rna", d, t),
        "mogonet": lambda d, t, c: fit_predict_mogonet(d, t, mogonet_cfg, v),
        "mogat": lambda d, t, c: fit_predict_mogat(d, t, mogat_cfg, v),
        "bionexus": lambda d, t, c: fit_predict_fused("graphgps", d, t, c, v),
        "bionexus_gat": lambda d, t, c: fit_predict_fused("gat", d, t, c, v),
        "bionexus_gcn": lambda d, t, c: fit_predict_fused("gcn", d, t, c, v),
        "bionexus_gine_local": lambda d, t, c: fit_predict_fused("gine", d, t, c, v),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["all"], help="method names, or 'all'")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--protocol", choices=["inner_val", "eval_fold"], default="inner_val")
    p.add_argument("--inner-val-frac", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--max-epochs", type=int, default=300, help="fused-model epochs (train.py default 300)")
    p.add_argument("--mogonet-pretrain-epochs", type=int, default=500)
    p.add_argument("--mogonet-epochs", type=int, default=2500)
    p.add_argument("--mogonet-weighting", choices=["official", "balanced"], default="official")
    p.add_argument("--mogat-graph", choices=["paper", "bionexus"], default="paper")
    p.add_argument("--mogat-lrs", nargs="+", type=float, default=[1e-3])
    p.add_argument("--mogat-max-epochs", type=int, default=500)
    p.add_argument("--mogat-min-epochs", type=int, default=200)
    p.add_argument("--mogat-search-iter", type=int, default=8)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4], help="repo root")
    p.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="LEAKAGE CHECK: randomly permute labels. Every method must then score at chance; "
        "results go to <protocol>_SHUFFLED so they never mix with real results.",
    )
    p.add_argument("--force", action="store_true", help="re-run even if a result file exists")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    registry = _build_registry(args)
    methods = list(registry) if args.methods == ["all"] else args.methods
    unknown = [m for m in methods if m not in registry]
    if unknown:
        raise SystemExit(f"unknown method(s): {unknown}. Choose from {list(registry)}")

    if args.protocol == "eval_fold":
        print(
            "[warning] protocol=eval_fold early-stops on the SAME fold it reports (as train.py does). "
            "Use inner_val for anything you publish."
        )

    data = DataBundle(*load_data(args.root))
    proto_dir = args.protocol
    if args.shuffle_labels:
        data.y = np.random.default_rng(0).permutation(data.y)
        proto_dir = f"{args.protocol}_SHUFFLED"
        print("[leakage check] labels randomly permuted - expect chance-level scores from every method")
    y = data.y
    out_root = args.root / "data" / "processed" / "results" / "baselines"
    print(f"{len(y)} patients | protocol={args.protocol} | methods={methods} | seeds={args.seeds}")

    for seed in args.seeds:
        folds = outer_folds(y, args.n_folds, seed)
        for method in methods:
            path = result_path(out_root, proto_dir, method, seed)
            if path.exists() and not args.force:
                print(f"[skip] {method} seed={seed} (exists: {path.name})")
                continue

            print(f"\n=== {method} | seed {seed} | {args.n_folds}-fold | {args.protocol} ===")
            fold_records, oof_pred = [], np.full(len(y), -1, dtype=int)
            t_method = time.time()
            for fold, (train_idx, test_idx) in enumerate(folds, start=1):
                set_all_seeds(seed * 1000 + fold)
                task = make_task(y, train_idx, test_idx, seed, fold, args.protocol, args.inner_val_frac)
                train_cfg = TrainConfig(n_folds=args.n_folds, max_epochs=args.max_epochs, seed=seed)
                t0 = time.time()
                out = registry[method](data, task, train_cfg)
                pred = np.asarray(out.pop("pred"))
                oof_pred[task.test_idx] = pred
                m = compute_metrics(y[task.test_idx], pred)
                secs = time.time() - t0
                print(
                    f"  fold {fold}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                    f"wF1={m['weighted_f1']:.4f}  ({secs:.0f}s)"
                )
                fold_records.append({"fold": fold, **m, "seconds": secs, "extras": out})

            assert (oof_pred >= 0).all(), "some patients never appeared in a test fold"
            accs = [r["accuracy"] for r in fold_records]
            f1s = [r["macro_f1"] for r in fold_records]
            print(
                f"  -> accuracy {np.mean(accs):.4f} +/- {np.std(accs):.4f} | "
                f"macroF1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  ({time.time() - t_method:.0f}s total)"
            )
            save_result(
                path,
                {
                    "method": method,
                    "seed": seed,
                    "protocol": args.protocol,
                    "n_folds": args.n_folds,
                    "inner_val_frac": args.inner_val_frac,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "folds": fold_records,
                    "y_true": y,
                    "y_pred": oof_pred,
                },
            )

    print(f"\nDone. Results in {out_root / proto_dir}. Next: python -m bio_nexus.evaluation.baselines.compare --protocol {proto_dir}")


if __name__ == "__main__":
    main()
