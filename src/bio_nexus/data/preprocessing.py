"""
Stage 1 — Multi-Omics Data Preprocessing & Cohort Building.

Owner: ML side / Data Engineering.
Provides modular functions for:
  - Building patient cohort mappings from GDC sample sheets.
  - Parsing and aligning multi-omics files (RNA-seq, miRNA, Methylation).
  - Platform detection (HM450 vs HM27 methylation arrays).
  - Log2 normalization, missing value imputation, and variance filtering.
  - Extracting PAM50 subtype labels and locking in the final patient cohort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_filename_index(data_dir: Path) -> dict[str, Path]:
    """Index files in data_dir mapping filename -> full path."""
    index = {}
    if data_dir.exists():
        for fpath in data_dir.rglob("*"):
            if fpath.is_file():
                index[fpath.name] = fpath
    return index


def count_lines(fpath: Path) -> int:
    """Fast line count reading raw bytes."""
    with open(fpath, "rb") as f:
        return f.read().count(b"\n")


def read_rna_file(fpath: Path) -> pd.Series:
    """Read GDC STAR-Counts RNA-seq file."""
    df = pd.read_csv(
        fpath, sep="\t", skiprows=[0, 2, 3, 4, 5], header=0, usecols=["gene_id", "tpm_unstranded"]
    )
    return df.set_index("gene_id")["tpm_unstranded"]


def read_mirna_file(fpath: Path) -> pd.Series:
    """Read GDC miRNA quantification file."""
    df = pd.read_csv(fpath, sep="\t", usecols=["miRNA_ID", "read_count"])
    return df.set_index("miRNA_ID")["read_count"]


def load_matrix(
    mapping: pd.DataFrame, file_col: str, data_dir: Path, layer_name: str, file_reader
) -> pd.DataFrame:
    """Build patients x features matrix for an omics layer."""
    file_index = build_filename_index(data_dir)
    rows = {}
    for _, row in mapping.iterrows():
        case_id = row["Case ID"]
        fname = row[file_col]
        fpath = file_index.get(fname)
        if fpath is None:
            continue
        try:
            rows[case_id] = file_reader(fpath)
        except Exception as e:
            print(f"[{layer_name}] WARNING: failed to read {fpath.name} for {case_id}: {e}")

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix.index.name = "patient_id"
    return matrix


def load_methylation_matrix(mapping: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Build methylation matrix filtered to dominant array platform (e.g. HM450)."""
    file_index = build_filename_index(data_dir)
    case_files = []
    for _, row in mapping.iterrows():
        fpath = file_index.get(row["meth_file"])
        if fpath is None:
            continue
        case_files.append((row["Case ID"], fpath))

    if not case_files:
        raise RuntimeError("[meth] No methylation files found.")

    line_counts = {}
    for case_id, fpath in case_files:
        line_counts[case_id] = count_lines(fpath)

    counts_series = pd.Series(line_counts)
    dominant_n_lines = counts_series.value_counts().index[0]

    kept_case_files = [
        (cid, fpath) for cid, fpath in case_files if line_counts[cid] == dominant_n_lines
    ]

    ref_case_id, ref_fpath = kept_case_files[0]
    ref = pd.read_csv(ref_fpath, sep="\t", header=None, names=["probe_id", "beta"], usecols=[0, 1])
    probe_ids = ref["probe_id"].to_numpy()
    n_probes = len(probe_ids)

    values = np.full((len(kept_case_files), n_probes), np.nan, dtype=np.float32)
    case_ids: list[str] = []

    for case_id, fpath in kept_case_files:
        try:
            beta = (
                pd.read_csv(fpath, sep="\t", header=None, usecols=[1])
                .iloc[:, 0]
                .to_numpy(dtype=np.float32)
            )
            if len(beta) != n_probes:
                continue
            values[len(case_ids)] = beta
            case_ids.append(case_id)
        except Exception:
            continue

    values = values[: len(case_ids)]
    matrix = pd.DataFrame(values, index=case_ids, columns=probe_ids)
    matrix.index.name = "patient_id"
    return matrix


def normalize_rna_like(matrix: pd.DataFrame) -> pd.DataFrame:
    """log2(x + 1) normalization for RNA-seq / miRNA data."""
    return np.log2(matrix.fillna(0) + 1)


def clean_missing(matrix: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """Drop features exceeding missing threshold and median-impute remainder."""
    frac_missing = matrix.isna().mean(axis=0)
    keep_cols = frac_missing[frac_missing <= threshold].index
    matrix = matrix[keep_cols]
    return matrix.fillna(matrix.median(axis=0))


def variance_filter(matrix: pd.DataFrame, top_n: int = 5000) -> pd.DataFrame:
    """Keep top N highest-variance features."""
    variances = matrix.var(axis=0).sort_values(ascending=False)
    keep_cols = variances.head(top_n).index
    return matrix[keep_cols]
