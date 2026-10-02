import os, glob, json
import torch
import pandas as pd

print("=== A. CHECKPOINTS ===")
for p in glob.glob("**/*.pt", recursive=True) + glob.glob("**/*.pth", recursive=True):
    sz = os.path.getsize(p)
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            fld = ckpt.get("fold", "N/A")
            epoch = ckpt.get("epoch", "N/A")
            print(f"{p}: size={sz}, keys={keys[:5]}, fold={fld}, epoch={epoch}")
        else:
            print(f"{p}: size={sz}, type={type(ckpt)}")
    except Exception as e:
        print(f"{p}: size={sz}, load_err={e}")

print("\n=== B. MATRICES & FEATURES ===")
for name, p in [
    ("rna_matrix", "data/serving/rna_matrix.parquet"),
    ("mirna_matrix", "data/serving/mirna_matrix.parquet"),
    ("meth_matrix", "data/serving/meth_matrix.parquet"),
    ("rna_feat", "data/serving/rna_features.csv"),
    ("mirna_feat", "data/serving/mirna_features.csv"),
    ("meth_feat", "data/serving/meth_features.csv"),
]:
    if os.path.exists(p):
        if p.endswith(".parquet"):
            df = pd.read_parquet(p)
            print(f"{name}: shape={df.shape}")
        else:
            df = pd.read_csv(p)
            print(f"{name}: shape={df.shape}, cols={list(df.columns)}")
    else:
        print(f"{name}: NOT FOUND at {p}")

print("\n=== C. GRAPHS ===")
for name, p in [
    ("rna_graph", "data/processed/graphs/rna_graph.pt"),
    ("mirna_graph", "data/processed/graphs/mirna_graph.pt"),
    ("meth_graph", "data/processed/graphs/meth_graph.pt"),
    ("node_order", "data/processed/graphs/graph_node_order.csv"),
]:
    if os.path.exists(p):
        if p.endswith(".pt"):
            g = torch.load(p, map_location="cpu", weights_only=False)
            x_s = getattr(g, "x", None).shape if hasattr(g, "x") and g.x is not None else None
            e_s = g.edge_index.shape if hasattr(g, "edge_index") else None
            a_s = g.edge_attr.shape if hasattr(g, "edge_attr") and g.edge_attr is not None else None
            print(f"{name}: type={type(g)}, x={x_s}, edge_index={e_s}, edge_attr={a_s}")
        else:
            df = pd.read_csv(p)
            print(f"{name}: shape={df.shape}, head={df.iloc[:2].to_dict()}")
    else:
        print(f"{name}: NOT FOUND at {p}")

print("\n=== D. LABELS ===")
for p in ["data/processed/final/labels_final.csv", "data/serving/labels.json", "data/serving/patients.csv"]:
    if os.path.exists(p):
        if p.endswith(".csv"):
            df = pd.read_csv(p)
            print(f"{p}: shape={df.shape}, cols={list(df.columns)}")
        else:
            with open(p) as f:
                d = json.load(f)
                print(f"{p}: len={len(d)}")
    else:
        print(f"{p}: NOT FOUND")

print("\n=== E. CLINICAL DATA ===")
p = "data/brca_tcga_pan_can_atlas_2018_clinical_data.tsv"
if os.path.exists(p):
    df = pd.read_csv(p, sep="\t")
    print(f"{p}: shape={df.shape}, cols_count={len(df.columns)}")
    key_cols = [c for c in df.columns if any(k in c.lower() for k in ["age", "er_", "pr_", "her2", "grade", "stage", "os_", "survival", "histol", "vital"])]
    print(f"Key cols: {key_cols}")
else:
    print(f"{p}: NOT FOUND")

print("\n=== F. FOLDS IN OOF / LABELS ===")
p = "data/serving/oof_predictions.parquet"
if os.path.exists(p):
    df = pd.read_parquet(p)
    print(f"{p}: shape={df.shape}, cols={list(df.columns)}")
    if "fold" in df.columns:
        print(f"Fold counts in oof:\n{df['fold'].value_counts()}")
else:
    print(f"{p}: NOT FOUND")

print("\n=== H. GENE SETS (.gmt) ===")
gmt_files = glob.glob("**/*.gmt", recursive=True)
print(f"GMT found: {len(gmt_files)}")
for p in gmt_files:
    print("  ", p, os.path.getsize(p))
