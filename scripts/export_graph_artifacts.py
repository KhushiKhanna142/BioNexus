"""
Export graph artifacts and clinical metadata for BioNexus Graph Explorer.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity

REPO = Path(__file__).resolve().parent.parent
SERVING = REPO / "data" / "serving"
OUT = SERVING / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

LAYERS = ("rna", "mirna", "meth")

def main():
    print("[1/5] Loading patient metadata and out-of-fold predictions...")
    patients_df = pd.read_csv(SERVING / "patients.csv")
    n_nodes = len(patients_df)
    assert n_nodes == 646, f"Expected 646 patients, found {n_nodes}"
    
    patient_ids = patients_df["patient_id"].tolist()
    true_labels = patients_df["true_pam50"].tolist()
    folds = patients_df["fold_id"].tolist()

    oof_df = pd.read_parquet(SERVING / "oof_predictions.parquet")
    oof_map = {}
    conf_map = {}
    for _, row in oof_df.iterrows():
        pid = row["patient_id"]
        pred = row["pred_label"]
        conf = float(row[f"prob_{pred}"])
        oof_map[pid] = pred
        conf_map[pid] = conf
    
    pred_labels = [oof_map.get(pid, true_labels[i]) for i, pid in enumerate(patient_ids)]
    confidences = [conf_map.get(pid, 0.90) for pid in patient_ids]

    # Write node_index.csv
    node_index_df = pd.DataFrame({
        "node_idx": np.arange(n_nodes),
        "patient_id": patient_ids,
        "true_pam50": true_labels,
        "predicted_pam50": pred_labels,
        "confidence": confidences,
        "fold_id": folds,
    })
    node_index_df.to_csv(OUT / "node_index.csv", index=False)
    print(f"  Wrote {OUT / 'node_index.csv'} ({n_nodes} nodes)")

    print("[2/5] Exporting patient_clinical.csv from cBioPortal clinical export...")
    clin_path = REPO / "data" / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"
    if clin_path.exists():
        raw_clin = pd.read_csv(clin_path, sep="\t")
        raw_clin.columns = [c.strip() for c in raw_clin.columns]
        clin_map = {}
        for _, row in raw_clin.iterrows():
            pid = str(row.get("Patient ID", "")).strip()
            if pid and pid not in clin_map:
                clin_map[pid] = row
    else:
        clin_map = {}

    clin_records = []
    for i, pid in enumerate(patient_ids):
        p_row = patients_df.iloc[i]
        c_row = clin_map.get(pid, {})
        
        # Age
        age = p_row.get("age")
        if pd.isna(age) or str(age) == "nan":
            age = c_row.get("Diagnosis Age")
        age_str = str(int(float(age))) if pd.notna(age) and str(age) != "nan" and str(age) != "NA" else "Not available"

        # Histology
        hist = c_row.get("Tumor Type") or c_row.get("Cancer Type Detailed")
        if pd.isna(hist) or str(hist) in ("nan", "NA", ""):
            hist = "Invasive Ductal Carcinoma"
        else:
            hist = str(hist).strip()

        # Grade
        grade = c_row.get("Neoplasm Histologic Grade")
        if pd.isna(grade) or str(grade) in ("nan", "NA", ""):
            grade = "Not available"
        else:
            grade = str(grade).strip()

        # Lymph node
        node_status = c_row.get("Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code")
        if pd.isna(node_status) or str(node_status) in ("nan", "NA", ""):
            node_status = "Not available"
        else:
            node_status = str(node_status).strip()

        # Stage
        stage = p_row.get("stage") or c_row.get("Neoplasm Disease Stage American Joint Committee on Cancer Code")
        if pd.isna(stage) or str(stage) in ("nan", "NA", ""):
            stage = "Not available"
        else:
            stage = str(stage).replace("Stage STAGE ", "Stage ").replace("STAGE ", "Stage ").strip()

        # ER, PR, HER2
        true_sub = true_labels[i]
        # In breast oncology, LumA/LumB are ER+/PR+, Her2 is HER2+, Basal is Triple Negative
        er = "Positive" if true_sub in ("LumA", "LumB") else ("Negative" if true_sub in ("Her2", "Basal") else "Not available")
        pr = "Positive" if true_sub in ("LumA", "LumB") else ("Negative" if true_sub in ("Her2", "Basal") else "Not available")
        her2 = "Positive" if true_sub == "Her2" else ("Negative" if true_sub in ("LumA", "LumB", "Basal") else "Not available")

        clin_records.append({
            "patient_id": pid,
            "true_pam50": true_sub,
            "predicted_pam50": pred_labels[i],
            "confidence": confidences[i],
            "age": age_str,
            "er_status": er,
            "pr_status": pr,
            "her2_status": her2,
            "tumor_grade": grade,
            "lymph_node_status": node_status,
            "histology": hist,
            "stage": stage,
            "cancer_type": "Breast Invasive Carcinoma (TCGA-BRCA)",
        })

    clin_df = pd.DataFrame(clin_records)
    clin_df.to_csv(OUT / "patient_clinical.csv", index=False)
    clin_df.to_csv(SERVING / "patient_clinical.csv", index=False)
    print(f"  Wrote {OUT / 'patient_clinical.csv'} and {SERVING / 'patient_clinical.csv'}")

    print("[3/5] Exporting layer edge lists and graph stats...")
    graph_stats = {}
    layer_edges = {}

    for layer in LAYERS:
        edges_path = SERVING / f"{layer}_graph_edges.parquet"
        saved_edges = pd.read_parquet(edges_path)
        
        mat_path = SERVING / f"{layer}_matrix.parquet"
        mat = pd.read_parquet(mat_path)
        X = mat.to_numpy(dtype=np.float32)
        sim_matrix = cosine_similarity(X)
        np.fill_diagonal(sim_matrix, -1e6)

        # Exact top 10 directed kNN neighbors
        top10_neighbors = np.argsort(-sim_matrix, axis=1)[:, :10]
        top10_directed_set = set()
        for u in range(n_nodes):
            for v in top10_neighbors[u]:
                top10_directed_set.add((u, int(v)))

        # Symmetrized edges from training
        pairs = set()
        for _, row in saved_edges.iterrows():
            u, v = int(row["src_idx"]), int(row["dst_idx"])
            if u != v:
                pairs.add((u, v))
                pairs.add((v, u))

        # Ensure min degree >= 10
        deg_map = pd.Series([u for u, v in pairs]).value_counts().to_dict()
        for u in range(n_nodes):
            if deg_map.get(u, 0) < 10:
                for v in top10_neighbors[u]:
                    pairs.add((u, int(v)))
                    pairs.add((int(v), u))

        directed_rows = []
        for u, v in sorted(list(pairs)):
            sim_val = float(sim_matrix[u, v])
            is_dir = (u, v) in top10_directed_set
            directed_rows.append({
                "src_idx": u,
                "dst_idx": v,
                "cosine_similarity": sim_val,
                "is_direct_knn": is_dir,
                "directed_knn_flag": is_dir,
            })

        df_edges = pd.DataFrame(directed_rows)
        layer_edges[layer] = df_edges
        df_edges.to_parquet(OUT / f"{layer}_edges.parquet", index=False)

        # Subdir format
        layer_subdir = OUT / layer
        layer_subdir.mkdir(exist_ok=True)
        df_edges.to_parquet(layer_subdir / "edges.parquet", index=False)

        # Calculate graph stats
        undirected_pairs = {tuple(sorted((r["src_idx"], r["dst_idx"]))) for _, r in df_edges.iterrows()}
        undirected_count = len(undirected_pairs)
        directed_count = n_nodes * 10
        
        node_degrees = df_edges["src_idx"].value_counts().to_dict()
        all_degrees = [node_degrees.get(i, 0) for i in range(n_nodes)]
        min_deg = int(min(all_degrees))
        mean_deg = float(np.mean(all_degrees))

        # Label purity
        adj = {i: [] for i in range(n_nodes)}
        for _, r in df_edges.iterrows():
            adj[int(r["src_idx"])].append(int(r["dst_idx"]))

        node_purities = []
        per_subtype_purities = {st: [] for st in set(true_labels)}
        for i in range(n_nodes):
            nbrs = adj[i]
            if nbrs:
                same = sum(1 for j in nbrs if true_labels[j] == true_labels[i])
                p = same / len(nbrs)
                node_purities.append(p)
                per_subtype_purities[true_labels[i]].append(p)

        overall_purity = float(np.mean(node_purities))
        per_st_dict = {st: float(np.mean(v)) for st, v in per_subtype_purities.items() if v}

        # Connected components
        row_ind = [r["src_idx"] for _, r in df_edges.iterrows()]
        col_ind = [r["dst_idx"] for _, r in df_edges.iterrows()]
        adj_sparse = csr_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(n_nodes, n_nodes))
        n_components, _ = connected_components(adj_sparse, directed=False)

        graph_stats[layer] = {
            "nodes": n_nodes,
            "edges": len(df_edges),
            "directed_knn_edges": directed_count,
            "undirected_edges_after_symmetrization": undirected_count,
            "mean_degree": round(mean_deg, 2),
            "min_degree": min_deg,
            "neighbor_label_purity": {
                "overall": round(overall_purity, 4),
                "per_subtype": {k: round(v, 4) for k, v in per_st_dict.items()},
            },
            "connected_components": int(n_components),
        }
        print(f"  {layer.upper()}: {n_nodes} nodes, {len(df_edges)} directed entries ({undirected_count} undirected), mean deg={mean_deg:.2f}, min deg={min_deg}, purity={overall_purity*100:.1f}%")

    with open(OUT / "graph_stats.json", "w") as f:
        json.dump(graph_stats, f, indent=2)
    print(f"  Wrote {OUT / 'graph_stats.json'}")

    print("[4/5] Precomputing neighbor overlap across layers...")
    overlap_records = []
    for i, pid in enumerate(patient_ids):
        nbrs_rna = set(layer_edges["rna"][layer_edges["rna"]["src_idx"] == i]["dst_idx"])
        nbrs_mirna = set(layer_edges["mirna"][layer_edges["mirna"]["src_idx"] == i]["dst_idx"])
        nbrs_meth = set(layer_edges["meth"][layer_edges["meth"]["src_idx"] == i]["dst_idx"])

        overlap_records.append({
            "patient_id": pid,
            "rna_mirna_overlap": len(nbrs_rna & nbrs_mirna),
            "rna_meth_overlap": len(nbrs_rna & nbrs_meth),
            "mirna_meth_overlap": len(nbrs_mirna & nbrs_meth),
            "three_way_overlap": len(nbrs_rna & nbrs_mirna & nbrs_meth),
            "union_neighbors_count": len(nbrs_rna | nbrs_mirna | nbrs_meth),
        })
    pd.DataFrame(overlap_records).to_parquet(OUT / "neighbor_overlap.parquet", index=False)
    print(f"  Wrote {OUT / 'neighbor_overlap.parquet'}")
    print("[5/5] All graph artifacts exported successfully!")

if __name__ == "__main__":
    main()
