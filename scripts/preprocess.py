"""
BioNexus - Preprocessing Script (Step 1 proper)
--------------------------------------------------
Loads raw omics files for the final cohort (cohort_case_ids.txt), builds
one matrix per layer (rows = patients, columns = genes/probes/miRNAs),
normalizes each layer, handles missing values, and saves 3 clean
matrices + the shared patient index - ready for graph construction.

Usage:
    python preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---- EDIT THESE PATHS IF NEEDED ----
BASE_DIR = Path(__file__).resolve().parent.parent

RNA_DIR = BASE_DIR / "data" / "raw" / "rnaseq"
MIRNA_DIR = BASE_DIR / "data" / "raw" / "mirna"
METH_DIR = BASE_DIR / "data" / "raw" / "methylation"

MAPPING_PATH = BASE_DIR / "data" / "processed" / "patient_cohort_mapping.csv"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- TUNABLE PARAMS ----
METH_TOP_N_PROBES = 5000        # variance-filter methylation down to this many probes
RNA_TOP_N_GENES = 5000          # variance-filter RNA down to this many genes (was previously unfiltered -
                                 # 60660 raw genes vs 646 patients is a severe feature/sample ratio)
MISSING_COL_THRESHOLD = 0.20    # drop a feature if missing in more than this fraction of patients


def build_filename_index(data_dir: Path) -> dict:
    """Walk the directory ONCE and map filename -> full path, instead of
    doing a fresh recursive rglob scan per file (which is very slow
    across 1000+ nested GDC download folders)."""
    index = {}
    for fpath in data_dir.rglob("*"):
        if fpath.is_file():
            index[fpath.name] = fpath
    return index


def load_matrix(mapping: pd.DataFrame, file_col: str, data_dir: Path, layer_name: str,
                 file_reader) -> pd.DataFrame:
    """Build a patients x features matrix for one omics layer by reading
    each patient's file and stacking rows. file_reader takes a filepath
    and returns a pandas Series indexed by feature name."""
    print(f"[{layer_name}] indexing {data_dir} ...")
    file_index = build_filename_index(data_dir)
    print(f"[{layer_name}] found {len(file_index)} files on disk")

    rows = {}
    for _, row in mapping.iterrows():
        case_id = row["Case ID"]
        fname = row[file_col]
        fpath = file_index.get(fname)
        if fpath is None:
            print(f"[{layer_name}] WARNING: file not found for {case_id}: {fname}")
            continue
        try:
            rows[case_id] = file_reader(fpath)
        except Exception as e:
            print(f"[{layer_name}] WARNING: failed to read {fpath.name} for {case_id}: {e}")

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix.index.name = "patient_id"
    print(f"[{layer_name}] built matrix: {matrix.shape[0]} patients x {matrix.shape[1]} features")
    return matrix


def read_rna_file(fpath: Path) -> pd.Series:
    """GDC STAR-Counts RNA-seq file: tab-separated.
    Raw line layout:
      line 0: '# gene-model...' comment
      line 1: header (gene_id, gene_name, ..., tpm_unstranded, ...)
      lines 2-5: N_unmapped/N_multimapping/N_noFeature/N_ambiguous summary rows
      line 6+: actual per-gene data
    Explicitly skip lines 0 and 2-5, keep line 1 as the header."""
    df = pd.read_csv(fpath, sep="\t", skiprows=[0, 2, 3, 4, 5], header=0,
                      usecols=["gene_id", "tpm_unstranded"])
    return df.set_index("gene_id")["tpm_unstranded"]


def read_mirna_file(fpath: Path) -> pd.Series:
    """GDC miRNA quantification file: tab-separated, miRNA_ID + read_count."""
    df = pd.read_csv(fpath, sep="\t", usecols=["miRNA_ID", "read_count"])
    return df.set_index("miRNA_ID")["read_count"]


def count_lines(fpath: Path) -> int:
    """Fast line count - reads raw bytes and counts newlines in C,
    much faster than parsing the file with pandas just to get a row count."""
    with open(fpath, "rb") as f:
        return f.read().count(b"\n")


def load_methylation_matrix(mapping: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Methylation files are large (~450K rows) and TCGA-BRCA mixes TWO
    array platforms: HumanMethylation450 (~486K probes, most samples) and
    the older HumanMethylation27 (~27K probes, a small minority). Files on
    different platforms have different probe sets/order and can't be
    stacked together. Strategy:
      1. Fast pre-pass: count lines in every file (no parsing) to find
         which platform (row count) is dominant.
      2. Use a file from the DOMINANT platform as the reference probe order.
      3. Main pass: only parse files matching that row count, reading just
         the numeric beta column (skip re-parsing probe ID strings) and
         assigning by position. Files on the minority platform are dropped
         and reported."""
    print(f"[meth] indexing {data_dir} ...")
    file_index = build_filename_index(data_dir)
    print(f"[meth] found {len(file_index)} files on disk")

    case_files = []
    for _, row in mapping.iterrows():
        fpath = file_index.get(row["meth_file"])
        if fpath is None:
            print(f"[meth] WARNING: file not found for {row['Case ID']}: {row['meth_file']}")
            continue
        case_files.append((row["Case ID"], fpath))

    if not case_files:
        raise RuntimeError("[meth] no methylation files found - check METH_DIR / mapping")

    # ---- Pass 0: fast line-count scan to find the dominant platform ----
    print(f"[meth] scanning {len(case_files)} files to detect array platform (line counts)...")
    line_counts = {}
    for i, (case_id, fpath) in enumerate(case_files):
        line_counts[case_id] = count_lines(fpath)
        if (i + 1) % 200 == 0:
            print(f"[meth] scanned {i + 1}/{len(case_files)} files...")

    counts_series = pd.Series(line_counts)
    platform_counts = counts_series.value_counts()
    dominant_n_lines = platform_counts.index[0]
    print(f"[meth] platform line-count distribution:\n{platform_counts.to_string()}")
    print(f"[meth] using dominant platform: {dominant_n_lines} lines "
          f"({platform_counts.iloc[0]} of {len(case_files)} patients)")

    kept_case_files = [(cid, fpath) for cid, fpath in case_files if line_counts[cid] == dominant_n_lines]
    n_dropped = len(case_files) - len(kept_case_files)
    if n_dropped:
        print(f"[meth] dropping {n_dropped} patients on a different array platform "
              f"(different probe set - can't be stacked with the majority)")

    # ---- Reference probe order, from a file on the dominant platform ----
    ref_case_id, ref_fpath = kept_case_files[0]
    ref = pd.read_csv(ref_fpath, sep="\t", header=None, names=["probe_id", "beta"], usecols=[0, 1])
    probe_ids = ref["probe_id"].to_numpy()
    n_probes = len(probe_ids)
    print(f"[meth] reference probe order: {n_probes} probes (from {ref_fpath.name})")

    # ---- Main pass: read beta column only, assign by position ----
    values = np.full((len(kept_case_files), n_probes), np.nan, dtype=np.float32)
    case_ids = []

    for i, (case_id, fpath) in enumerate(kept_case_files):
        try:
            beta = pd.read_csv(fpath, sep="\t", header=None, usecols=[1]).iloc[:, 0].to_numpy(dtype=np.float32)
            if len(beta) != n_probes:
                print(f"[meth] WARNING: {case_id} has {len(beta)} rows, expected {n_probes} - skipped")
                continue
            values[len(case_ids)] = beta
            case_ids.append(case_id)
        except Exception as e:
            print(f"[meth] WARNING: failed to read {fpath.name} for {case_id}: {e}")

        if (i + 1) % 100 == 0:
            print(f"[meth] processed {i + 1}/{len(kept_case_files)} files...")

    values = values[:len(case_ids)]
    matrix = pd.DataFrame(values, index=case_ids, columns=probe_ids)
    matrix.index.name = "patient_id"
    print(f"[meth] built matrix: {matrix.shape[0]} patients x {matrix.shape[1]} features")
    return matrix


def normalize_rna_like(matrix: pd.DataFrame) -> pd.DataFrame:
    """log2(x + 1) normalization for RNA-seq / miRNA count-like data."""
    return np.log2(matrix.fillna(0) + 1)


def clean_missing(matrix: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    """Drop features missing in more than MISSING_COL_THRESHOLD of patients,
    then median-impute whatever's left."""
    frac_missing = matrix.isna().mean(axis=0)
    keep_cols = frac_missing[frac_missing <= MISSING_COL_THRESHOLD].index
    dropped = matrix.shape[1] - len(keep_cols)
    matrix = matrix[keep_cols]
    matrix = matrix.fillna(matrix.median(axis=0))
    print(f"[{layer_name}] dropped {dropped} features (>{int(MISSING_COL_THRESHOLD*100)}% missing), "
          f"median-imputed the rest -> {matrix.shape[1]} features remain")
    return matrix


def variance_filter(matrix: pd.DataFrame, top_n: int, layer_name: str) -> pd.DataFrame:
    variances = matrix.var(axis=0).sort_values(ascending=False)
    keep_cols = variances.head(top_n).index
    print(f"[{layer_name}] variance-filtered {matrix.shape[1]} -> {len(keep_cols)} probes")
    return matrix[keep_cols]


def main():
    mapping = pd.read_csv(MAPPING_PATH)
    print(f"Loaded cohort mapping: {len(mapping)} patients\n")

    # ---- Build raw matrices ----
    rna = load_matrix(mapping, "rna_file", RNA_DIR, "rna", read_rna_file)
    mirna = load_matrix(mapping, "mirna_file", MIRNA_DIR, "mirna", read_mirna_file)
    meth = load_methylation_matrix(mapping, METH_DIR)

    # ---- Missing value handling (rna/mirna only for now - meth reordered below) ----
    print()
    rna = clean_missing(rna, "rna")
    mirna = clean_missing(mirna, "mirna")

    # ---- Variance-filter methylation FIRST (cheap, one pass) before doing
    # expensive median-based missing-value cleanup on only 98% fewer columns ----
    print()
    meth = variance_filter(meth, METH_TOP_N_PROBES, "meth")
    meth = clean_missing(meth, "meth")

    # ---- Normalize ----
    print()
    rna = normalize_rna_like(rna)
    mirna = normalize_rna_like(mirna)
    # methylation beta values are already 0-1 scaled, no transform needed
    print("[rna]   log2(x+1) normalization applied")
    print("[mirna] log2(x+1) normalization applied")
    print("[meth]  beta values already 0-1 scaled, no transform applied")

    # ---- Variance-filter RNA AFTER log2 normalization (not before, unlike meth) ----
    # RNA's raw TPM variance is dominated by a handful of extremely highly-expressed
    # genes, which isn't necessarily the most biologically informative signal.
    # Selecting top-variance genes in LOG-NORMALIZED space is standard genomics
    # practice (e.g. Seurat's highly-variable-gene selection) and avoids that bias.
    # This also fixes a real problem: 60660 raw genes vs 646 patients meant the
    # encoder's first linear layer was a severely underdetermined projection.
    print()
    rna = variance_filter(rna, RNA_TOP_N_GENES, "rna")

    # ---- Align all 3 layers to the SAME shared patient index ----
    shared_patients = rna.index.intersection(mirna.index).intersection(meth.index)
    shared_patients = sorted(shared_patients)
    print(f"\nShared patient index across all 3 layers (post file-read): {len(shared_patients)} patients")

    rna = rna.loc[shared_patients]
    mirna = mirna.loc[shared_patients]
    meth = meth.loc[shared_patients]

    # ---- Save ----
    rna.to_csv(OUT_DIR / "rna_matrix.csv")
    mirna.to_csv(OUT_DIR / "mirna_matrix.csv")
    meth.to_csv(OUT_DIR / "meth_matrix.csv")
    pd.Series(shared_patients, name="patient_id").to_csv(
        OUT_DIR / "shared_patient_index.csv", index=False
    )

    print(f"\nSaved:")
    print(f"  {OUT_DIR / 'rna_matrix.csv'}   ({rna.shape[0]} x {rna.shape[1]})")
    print(f"  {OUT_DIR / 'mirna_matrix.csv'} ({mirna.shape[0]} x {mirna.shape[1]})")
    print(f"  {OUT_DIR / 'meth_matrix.csv'}  ({meth.shape[0]} x {meth.shape[1]})")
    print(f"  {OUT_DIR / 'shared_patient_index.csv'}")


if __name__ == "__main__":
    main()