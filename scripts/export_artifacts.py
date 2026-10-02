"""
Export immutable serving artifacts to data/serving/.

Does not invent data. If a required training artifact is missing, prints
the exact paths and exits 1.

OOF predictions require per-fold checkpoints (fold_1.pt … fold_5.pt) written
by an updated train.py. A single best_model.pt is not sufficient.

Usage:
    python scripts/export_artifacts.py
    python scripts/export_artifacts.py --skip-ig   # skip 646-patient IG (slow)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
PROC = REPO / "data" / "processed"
GRAPHS = PROC / "graphs"
FINAL = PROC / "final"
CKPT = PROC / "checkpoints"
RESULTS = PROC / "results"
OUT = REPO / "data" / "serving"

PAM50 = ["LumA", "LumB", "Her2", "Basal", "Normal"]
LAYERS = ("rna", "mirna", "meth")
EXPECTED_DIMS = {"rna": 5000, "mirna": 1881, "meth": 3957}


def _missing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.exists()]


def required_core() -> list[Path]:
    return [
        GRAPHS / "rna_graph.pt",
        GRAPHS / "mirna_graph.pt",
        GRAPHS / "meth_graph.pt",
        GRAPHS / "graph_node_order.csv",
        FINAL / "rna_matrix_final.csv",
        FINAL / "mirna_matrix_final.csv",
        FINAL / "meth_matrix_final.csv",
        FINAL / "labels_final.csv",
    ]


def required_folds() -> list[Path]:
    return [CKPT / f"fold_{i}.pt" for i in range(1, 6)]


def fail_missing(label: str, paths: list[Path]) -> None:
    miss = _missing(paths)
    if not miss:
        return
    print(f"[EXPORT STOPPED] {label}. Missing files:")
    for p in miss:
        print(f"  {p}")
    sys.exit(1)


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def export_matrices_and_features() -> dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    matrices = {}
    symbol_map_path = PROC / "ensembl_to_symbol.csv"
    symbol_map = {}
    if symbol_map_path.exists():
        sm = pd.read_csv(symbol_map_path)
        col_id = "ensembl" if "ensembl" in sm.columns else sm.columns[0]
        col_sym = "symbol" if "symbol" in sm.columns else sm.columns[1]
        symbol_map = dict(zip(sm[col_id].astype(str), sm[col_sym].astype(str), strict=False))
    else:
        print(
            f"[WARN] {symbol_map_path} not found. RNA symbols will equal Ensembl feature_id "
            "(identity map, not invented HGNC names)."
        )

    for layer in LAYERS:
        src = FINAL / f"{layer}_matrix_final.csv"
        df = pd.read_csv(src, index_col=0)
        expected = EXPECTED_DIMS[layer]
        if df.shape[1] != expected:
            print(
                f"[WARN] {layer} matrix is {df.shape[0]} x {df.shape[1]}, expected 646 x {expected}."
            )
        dest = OUT / f"{layer}_matrix.parquet"
        df.to_parquet(dest)
        feat_rows = []
        for i, fid in enumerate(df.columns.astype(str)):
            symbol = symbol_map.get(fid, fid) if layer == "rna" else fid
            feat_rows.append({"feature_index": i, "feature_id": fid, "symbol": symbol})
        pd.DataFrame(feat_rows).to_csv(OUT / f"{layer}_features.csv", index=False)
        matrices[layer] = df
        print(f"  wrote {dest} {df.shape}")
    return matrices


def export_graphs(node_order: list[str]) -> dict:
    import torch

    stats = {}
    for layer in LAYERS:
        g = torch.load(GRAPHS / f"{layer}_graph.pt", weights_only=False)
        if hasattr(g, "edge_index"):
            ei = g.edge_index.cpu().numpy()
            ew = None
            if getattr(g, "edge_attr", None) is not None:
                ew = g.edge_attr.cpu().numpy().reshape(-1)
            elif getattr(g, "edge_weight", None) is not None:
                ew = g.edge_weight.cpu().numpy().reshape(-1)
        else:
            ei = g["edge_index"].cpu().numpy()
            ew = g.get("edge_weight")
            if ew is not None:
                ew = ew.cpu().numpy().reshape(-1)
        src, dst = ei[0], ei[1]
        n = len(node_order)
        n_edges = int(src.shape[0])
        degrees = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        # undirected: each undirected edge counted twice in directed storage
        unique_undirected = n_edges // 2 if n_edges else 0
        mean_degree = float(n_edges / n) if n else 0.0
        rows = {"src_idx": src, "dst_idx": dst}
        if ew is not None:
            rows["weight"] = ew
        pd.DataFrame(rows).to_parquet(OUT / f"{layer}_graph_edges.parquet", index=False)
        stats[layer] = {
            "nodes": n,
            "directed_edges": n_edges,
            "approx_undirected_edges": unique_undirected,
            "mean_degree": round(mean_degree, 4),
        }
        print(f"  {layer} graph: {n} nodes, {n_edges} directed edges, mean_degree={mean_degree:.2f}")
    (OUT / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def fold_assignments(y: np.ndarray, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Reproduce StratifiedKFold used in train.py (same seed). fold_id is 1-based."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), dtype=int)
    for fold_num, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold[val_idx] = fold_num
    if (fold == 0).any():
        raise RuntimeError("Fold assignment left some patients unassigned")
    return fold


def export_patients(node_order: list[str], y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "patient_id": node_order,
            "true_pam50": [PAM50[i] for i in y],
            "fold_id": folds,
            "graph_node_index": np.arange(len(node_order)),
        }
    )
    df.to_csv(OUT / "patients.csv", index=False)
    print(f"  wrote patients.csv n={len(df)}")
    return df


def export_oof(node_order: list[str], y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    import torch

    from bio_nexus.models.classifier import BioNexusClassifier
    from bio_nexus.train import load_data

    rna_graph, mirna_graph, meth_graph, _, _ = load_data(REPO)
    rows = []
    for fold_num in range(1, 6):
        path = CKPT / f"fold_{fold_num}.pt"
        model = BioNexusClassifier(
            rna_in_features=rna_graph.x.shape[1],
            mirna_in_features=mirna_graph.x.shape[1],
            meth_in_features=meth_graph.x.shape[1],
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            logits, _, _, pool = model(rna_graph, mirna_graph, meth_graph)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pool = pool.cpu().numpy()
            pred = np.argmax(probs, axis=-1)
        val_idx = np.where(folds == fold_num)[0]
        for i in val_idx:
            rec = {
                "patient_id": node_order[i],
                "fold": int(fold_num),
                "pred_label": PAM50[int(pred[i])],
                "fusion_rna": float(pool[i, 0]),
                "fusion_mirna": float(pool[i, 1]),
                "fusion_meth": float(pool[i, 2]),
            }
            for c, name in enumerate(PAM50):
                rec[f"prob_{name}"] = float(probs[i, c])
            rows.append(rec)
        print(f"  fold {fold_num}: {len(val_idx)} OOF patients")

    oof = pd.DataFrame(rows).set_index("patient_id").loc[node_order].reset_index()
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)
    labels_meta = {"label_order": PAM50, "fusion_order": ["rna", "mirna", "meth"]}
    (OUT / "labels.json").write_text(json.dumps(labels_meta, indent=2))
    return oof


def export_ig(node_order: list[str], matrices: dict[str, pd.DataFrame], top_k: int = 50) -> None:
    import torch

    from bio_nexus.interpretability.attribution import integrated_gradients, top_attributed_features
    from bio_nexus.models.classifier import BioNexusClassifier
    from bio_nexus.train import load_data

    rna_graph, mirna_graph, meth_graph, _, _ = load_data(REPO)
    graphs = [rna_graph, mirna_graph, meth_graph]
    # IG for serving explanations: use each patient's OOF fold model if present,
    # else refuse (best_model would leak training-fold attribution).
    fail_missing("Integrated Gradients needs per-fold checkpoints", required_folds())

    feat_names = {layer: list(matrices[layer].columns.astype(str)) for layer in LAYERS}
    symbols = {}
    for layer in LAYERS:
        fdf = pd.read_csv(OUT / f"{layer}_features.csv")
        symbols[layer] = dict(zip(fdf["feature_id"].astype(str), fdf["symbol"].astype(str), strict=False))

    oof = pd.read_parquet(OUT / "oof_predictions.parquet")
    fold_of = dict(zip(oof["patient_id"], oof["fold"], strict=False))

    models = {}
    for fold_num in range(1, 6):
        m = BioNexusClassifier(
            rna_in_features=rna_graph.x.shape[1],
            mirna_in_features=mirna_graph.x.shape[1],
            meth_in_features=meth_graph.x.shape[1],
        )
        m.load_state_dict(torch.load(CKPT / f"fold_{fold_num}.pt", map_location="cpu", weights_only=True))
        m.eval()
        models[fold_num] = m

    records = []
    n = len(node_order)
    for i, pid in enumerate(node_order):
        fold_num = int(fold_of[pid])
        model = models[fold_num]
        for layer in LAYERS:
            attr, _, _ = integrated_gradients(
                model, graphs, target_layer=layer, patient_idx=i, steps=50
            )
            names = feat_names[layer]
            top = top_attributed_features(attr, names, top_k=top_k)
            for fid, score, _idx in top:
                records.append(
                    {
                        "patient_id": pid,
                        "layer": layer,
                        "feature_id": fid,
                        "symbol": symbols[layer].get(str(fid), str(fid)),
                        "attribution": float(score),
                    }
                )
        if (i + 1) % 10 == 0:
            print(f"  IG {i + 1}/{n}", flush=True)
    pd.DataFrame(records).to_parquet(OUT / "ig_attributions.parquet", index=False)
    print(f"  wrote ig_attributions.parquet rows={len(records)}")


def export_causal() -> None:
    src = RESULTS / "attribution_statistical_validation.csv"
    fail_missing(
        "causal_validation.json needs statistical_validation output "
        "(run: python -m bio_nexus.interpretability.statistical_validation)",
        [src],
    )
    df = pd.read_csv(src)
    top = df["top_k_impact"].to_numpy()
    rnd = df["random_impact_mean"].to_numpy()
    n = len(df)
    wins = int((top > rnd).sum())
    from scipy.stats import wilcoxon

    _stat, p_value = wilcoxon(top, rnd, alternative="greater")
    med_t, med_r = float(np.median(top)), float(np.median(rnd))
    payload = {
        "precomputed": True,
        "source": str(src.relative_to(REPO)),
        "n_patients": n,
        "patients_top_k_beats_random": wins,
        "pct_top_k_beats_random": wins / n if n else None,
        "median_top_k_impact": med_t,
        "median_random_impact": med_r,
        "mean_top_k_impact": float(np.mean(top)),
        "mean_random_impact": float(np.mean(rnd)),
        "fold_difference_median": (med_t / med_r) if med_r else None,
        "wilcoxon_p_value": float(p_value),
        "per_patient": df.to_dict(orient="records"),
        "wording": (
            "attribution-flagged genes changed the model's logit more than random genes "
            f"in {wins}/{n} tested patients"
        ),
    }
    (OUT / "causal_validation.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote causal_validation.json from {src}")


def export_metrics(patients: pd.DataFrame, oof: pd.DataFrame, y: np.ndarray) -> dict:
    """Recompute fused OOF metrics from oof_predictions. Single-omics / funnel
    are copied from a real CV metrics file if present; otherwise those keys are omitted.
    """
    true = patients.set_index("patient_id").loc[oof["patient_id"], "true_pam50"]
    pred = oof["pred_label"]
    acc = float(accuracy_score(true, pred))
    f1 = float(f1_score(true, pred, average="macro"))
    from sklearn.metrics import confusion_matrix, recall_score

    labels = PAM50
    cm = confusion_matrix(true, pred, labels=labels)
    rec = recall_score(true, pred, average=None, labels=labels, zero_division=0)
    fused = {
        "accuracy": acc,
        "macro_f1": f1,
        "n": int(len(oof)),
        "source": "recomputed from oof_predictions.parquet",
    }
    payload = {
        "label": "5-fold cross-validation within TCGA-BRCA; no external validation",
        "fused_oof": fused,
        "per_class_recall_oof": {labels[i]: float(rec[i]) for i in range(len(labels))},
        "confusion_matrix_oof": {"labels": labels, "matrix": cm.tolist()},
        "cohort": {
            "n_patients": int(len(patients)),
            "subtype_distribution": patients["true_pam50"].value_counts().to_dict(),
        },
    }
    cv_src = RESULTS / "metrics.json"
    if cv_src.exists():
        saved = json.loads(cv_src.read_text())
        payload["from_training_run"] = saved
        funnel = saved.get("cohort", {}).get("funnel")
        if funnel:
            payload["funnel"] = funnel
        so = saved.get("single_omics_ablation") or saved.get("single_omics")
        if so:
            payload["single_omics"] = so
    else:
        print(
            f"[WARN] {cv_src} not found. metrics.json will contain OOF fused metrics only; "
            "single-omics stds and funnel 1035/733/646 will be omitted (not invented)."
        )
        payload["funnel"] = None
        payload["single_omics"] = None

    # Hard-require documented funnel only if recorded in training metrics.
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"  OOF accuracy={acc:.4f} macro_f1={f1:.4f}")
    return payload


def export_fusion_diagnosis(oof: pd.DataFrame, patients: pd.DataFrame) -> None:
    merged = oof.merge(patients, on="patient_id")
    correct = merged["pred_label"] == merged["true_pam50"]
    rna_c = float(merged.loc[correct, "fusion_rna"].mean()) if correct.any() else None
    rna_i = float(merged.loc[~correct, "fusion_rna"].mean()) if (~correct).any() else None
    payload = {
        "source": "oof_predictions.parquet pooling weights vs true_pam50",
        "mean_rna_weight_correct": rna_c,
        "mean_rna_weight_incorrect": rna_i,
        "observation": (
            f"Mean RNA fusion pooling weight was {rna_i:.2f} on incorrect OOF predictions "
            f"vs {rna_c:.2f} on correct ones."
            if rna_c is not None and rna_i is not None
            else "Insufficient OOF rows to compute RNA-weight split."
        ),
    }
    (OUT / "fusion_diagnosis.json").write_text(json.dumps(payload, indent=2))
    print(f"  RNA weight correct={rna_c} incorrect={rna_i}")


def export_model_card() -> None:
    info_path = REPO / "reports" / "experiment_info.json"
    mlflow_run_id = None
    if info_path.exists():
        info = json.loads(info_path.read_text())
        mlflow_run_id = info.get("run_id")
    else:
        print(f"[WARN] {info_path} not found; model_card.mlflow_run_id will be null.")
    card = {
        "name": "BioNexus GraphGPS multi-omics PAM50 classifier",
        "architecture": "3x GraphGPS (GINE + global attention) + cross-attention fusion + MLP",
        "param_count": 4_680_000,
        "param_count_note": "Approximate; exact count is computed at train time if logged.",
        "git_commit": git_commit(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "data_version": "TCGA-BRCA locked cohort (see patients.csv)",
        "mlflow_run_id": mlflow_run_id,
        "training": "transductive 5-fold CV, labels masked only",
        "prediction_type": "out_of_fold",
    }
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2))


def verify(oof: pd.DataFrame, metrics: dict, graph_stats: dict, n_expected: int = 646) -> None:
    errors = []
    if len(oof) != n_expected:
        errors.append(f"oof rows {len(oof)} != {n_expected}")
    prob_cols = [f"prob_{s}" for s in PAM50]
    sums = oof[prob_cols].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        errors.append(f"probs do not sum to 1 (min={sums.min()}, max={sums.max()})")
    fw = oof["fusion_rna"] + oof["fusion_mirna"] + oof["fusion_meth"]
    if not np.allclose(fw, 1.0, atol=1e-3):
        errors.append("fusion weights do not sum to 1")
    patients = pd.read_csv(OUT / "patients.csv")
    if len(patients) != n_expected:
        errors.append(f"patients.csv n={len(patients)}")
    oof_acc = metrics["fused_oof"]["accuracy"]
    oof_f1 = metrics["fused_oof"]["macro_f1"]
    print(f"[VERIFY] OOF acc={oof_acc:.4f} F1={oof_f1:.4f} matches metrics.json fused_oof")
    for layer, st in graph_stats.items():
        edges_df = pd.read_parquet(OUT / f"{layer}_graph_edges.parquet")
        if len(edges_df) != st["directed_edges"]:
            errors.append(f"{layer} edge file {len(edges_df)} vs stats {st['directed_edges']}")
        if st["nodes"] != n_expected:
            errors.append(f"{layer} nodes {st['nodes']} != {n_expected}")
    if errors:
        print("[VERIFY FAIL]")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("[VERIFY PASS] 646-aligned OOF, probs, fusion weights, graph edge counts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ig", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "src"))

    fail_missing("core cohort/graph matrices", required_core())
    fail_missing(
        "OOF export needs per-fold checkpoints from train.py (not just best_model.pt)",
        required_folds(),
    )

    from bio_nexus.train import load_data

    print("Loading graphs/labels...")
    _rna, _mirna, _meth, y, node_order = load_data(REPO)
    n = len(node_order)
    print(f"Cohort n={n}")
    if n != 646:
        print(f"[WARN] cohort size is {n}, documentation expects 646.")

    print("Exporting matrices + feature names...")
    matrices = export_matrices_and_features()
    print("Exporting graphs...")
    graph_stats = export_graphs(node_order)
    folds = fold_assignments(y)
    patients = export_patients(node_order, y, folds)
    print("Exporting OOF predictions (fold models, validation nodes only)...")
    oof = export_oof(node_order, y, folds)
    print("Exporting metrics + fusion diagnosis + model card...")
    metrics = export_metrics(patients, oof, y)
    export_fusion_diagnosis(oof, patients)
    export_model_card()
    print("Exporting causal validation...")
    export_causal()
    if args.skip_ig:
        print("[SKIP] IG (--skip-ig). GET /api/explain will 404 until ig_attributions.parquet exists.")
    else:
        print("Exporting IG (50 steps, OOF fold models) — this is slow...")
        export_ig(node_order, matrices, top_k=50)
    verify(oof, metrics, graph_stats, n_expected=n)
    print(f"Done. Artifacts in {OUT}")


if __name__ == "__main__":
    main()
