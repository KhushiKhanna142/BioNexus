"""
bio_nexus.evaluation.baselines.compare
--------------------------------------
Aggregates the JSON files written by run.py into
  summary.csv            mean +/- std per method (over all seed x fold results)
  paired_tests.csv       reference method vs every other method, paired on (seed, fold)
  per_class_recall.csv   recall per PAM50 subtype from pooled out-of-fold predictions

Usage:
  python -m bio_nexus.evaluation.baselines.compare --protocol inner_val --ref bionexus

STATISTICS - read before quoting a p-value:
  * Fold scores from k-fold CV share training data, so they are NOT independent.
    The plain paired t-test / Wilcoxon over folds is anti-conservative. The column
    `p_corrected_t` uses the Nadeau & Bengio (2003) corrected resampled t-test, the
    standard fix for repeated k-fold CV. Quote that one, with Holm adjustment
    (`p_corrected_t_holm`) because you compare against several methods.
  * With a single seed there are only 5 paired folds; a two-sided Wilcoxon test
    cannot go below p = 0.0625 at n = 5 no matter how large the gap. Run >= 5 seeds
    (25 pairs) before leaning on any test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bio_nexus.models.classifier import PAM50_SUBTYPES

from .common import METRICS


def load_results(root: Path, protocol: str):
    rows, oof = [], {}
    files = sorted((root / protocol).glob("*__seed*.json"))
    if not files:
        raise SystemExit(f"no result files in {root / protocol}. Run run.py first.")
    for p in files:
        d = json.loads(p.read_text())
        for r in d["folds"]:
            rows.append({"method": d["method"], "seed": d["seed"], "fold": r["fold"], **{k: r[k] for k in METRICS}})
        oof[(d["method"], d["seed"])] = (np.array(d["y_true"]), np.array(d["y_pred"]))
    return pd.DataFrame(rows), oof


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def corrected_resampled_t(d: np.ndarray, test_train_ratio: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003). d = per-(seed,fold) score differences."""
    j = len(d)
    var = d.var(ddof=1)
    if j < 2 or var == 0:
        return float("nan"), 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1.0 / j + test_train_ratio) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def summarise(df: pd.DataFrame, oof: dict) -> pd.DataFrame:
    out = []
    for method, g in df.groupby("method"):
        row = {"method": method, "n_results": len(g)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else 0.0
        # macro-F1 on pooled out-of-fold predictions (more stable for the rare Normal class)
        pooled = []
        for (meth, _seed), (yt, yp) in oof.items():
            if meth == method:
                from sklearn.metrics import f1_score

                pooled.append(f1_score(yt, yp, average="macro", zero_division=0))
        row["pooled_oof_macro_f1"] = float(np.mean(pooled)) if pooled else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, ref: str, n_folds: int, metric: str) -> pd.DataFrame:
    if ref not in set(df["method"]):
        raise SystemExit(f"reference method '{ref}' has no results")
    piv = df.pivot_table(index=["seed", "fold"], columns="method", values=metric)
    ratio = 1.0 / (n_folds - 1)  # n_test / n_train for k-fold
    rows = []
    for other in piv.columns:
        if other == ref:
            continue
        pair = piv[[ref, other]].dropna()
        if len(pair) < 2:
            continue
        d = (pair[ref] - pair[other]).to_numpy()
        try:
            w_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            w_p = float("nan")
        t, p = corrected_resampled_t(d, ratio)
        rows.append(
            {
                "ref": ref,
                "other": other,
                "metric": metric,
                "n_pairs": len(d),
                "mean_diff(ref-other)": d.mean(),
                "ref_wins": int((d > 0).sum()),
                "other_wins": int((d < 0).sum()),
                "p_wilcoxon_uncorrected": w_p,
                "t_corrected": t,
                "p_corrected_t": p,
            }
        )
    res = pd.DataFrame(rows)
    if len(res):
        res["p_corrected_t_holm"] = holm(res["p_corrected_t"].tolist())
    return res


def per_class_recall(oof: dict) -> pd.DataFrame:
    acc: dict[str, np.ndarray] = {}
    for (method, _seed), (yt, yp) in oof.items():
        cm = np.zeros((len(PAM50_SUBTYPES), len(PAM50_SUBTYPES)), dtype=int)
        for a, b in zip(yt, yp, strict=True):
            cm[a, b] += 1
        acc[method] = acc.get(method, 0) + cm
    rows = []
    for method, cm in acc.items():
        rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
        rows.append({"method": method, **{c: float(r) for c, r in zip(PAM50_SUBTYPES, rec, strict=True)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="inner_val", help="inner_val | eval_fold | inner_val_SHUFFLED | ...")
    ap.add_argument("--ref", default="bionexus")
    ap.add_argument("--metric", default="macro_f1", choices=list(METRICS))
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = ap.parse_args()

    res_root = args.root / "data" / "processed" / "results" / "baselines"
    df, oof = load_results(res_root, args.protocol)
    n_folds = int(df["fold"].max())
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = {"float_format": lambda v: f"{v:.4f}"}

    summ = summarise(df, oof)
    tests = paired_tests(df, args.ref, n_folds, args.metric)
    recall = per_class_recall(oof)

    out_dir = res_root / args.protocol
    summ.to_csv(out_dir / "summary.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)
    recall.to_csv(out_dir / "per_class_recall.csv", index=False)

    n_seeds = df["seed"].nunique()
    print(f"\nprotocol={args.protocol} | {n_seeds} seed(s) x {n_folds} folds | std is over all seed-fold results\n")
    print("== Summary ==")
    print(summ.to_string(index=False, **fmt))
    print(f"\n== Paired tests: {args.ref} vs others on {args.metric} (quote p_corrected_t_holm) ==")
    print(tests.to_string(index=False, **fmt) if len(tests) else "(no overlapping results)")
    print("\n== Per-class recall (pooled out-of-fold, summed over seeds) ==")
    print(recall.to_string(index=False, **fmt))
    if n_seeds < 5:
        print("\n[note] fewer than 5 seeds: treat all p-values as indicative only.")
    print(f"\nSaved summary.csv, paired_tests.csv, per_class_recall.csv to {out_dir}")


if __name__ == "__main__":
    main()
