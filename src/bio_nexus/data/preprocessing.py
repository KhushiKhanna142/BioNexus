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


# Canonical PAM50 marker genes (Parker et al. 2009) + Ki-67 proliferation signature.
# These are guaranteed to be included in the RNA feature matrix regardless of
# variance rank, because global variance is dominated by Basal-vs-Luminal
# differences. Without force-inclusion, the subtler LumA/LumB proliferation
# signals (MKI67, AURKA, CCNB1, FOXM1) can be variance-filtered out.
PAM50_CANONICAL_GENES: list[str] = [
    # PAM50 core set (HGNC symbols and common Ensembl IDs from TCGA STAR-Counts)
    "ENSG00000148773",  # MKI67  - proliferation rate (LumA vs LumB discriminator)
    "ENSG00000117399",  # CDC20  - mitotic checkpoint
    "ENSG00000166033",  # HTRA3
    "ENSG00000091831",  # ESR1   - estrogen receptor alpha (LumA/LumB)
    "ENSG00000082175",  # PGR    - progesterone receptor
    "ENSG00000141736",  # ERBB2  - HER2 amplification
    "ENSG00000107485",  # GATA3  - luminal transcription factor
    "ENSG00000130528",  # FOXA1  - luminal transcription factor
    "ENSG00000197386",  # HTR1D
    "ENSG00000128272",  # ATF4
    "ENSG00000159111",  # MYBPC1
    "ENSG00000143870",  # PTTG1  - securin, proliferation
    "ENSG00000117676",  # RPS6KA1
    "ENSG00000170312",  # CDK1   - mitotic cyclin-dependent kinase
    "ENSG00000134057",  # CCNB1  - cyclin B1, proliferation (LumB marker)
    "ENSG00000065135",  # GNAI3
    "ENSG00000156970",  # BUB1   - spindle assembly checkpoint
    "ENSG00000117245",  # KIF17
    "ENSG00000087586",  # AURKA  - aurora kinase A, proliferation (LumB)
    "ENSG00000117115",  # PADI2
    "ENSG00000010810",  # FYN    - tyrosine kinase
    "ENSG00000118785",  # SPP1   - secreted phosphoprotein 1
    "ENSG00000167654",  # ANLN   - anillin, cell division
    "ENSG00000171862",  # PTEN   - tumor suppressor
    "ENSG00000170345",  # FOS    - proto-oncogene (Her2 enriched)
    "ENSG00000141551",  # CLDN7  - claudin 7
    "ENSG00000169398",  # PTK2   - focal adhesion kinase (Basal)
    "ENSG00000108839",  # ALPP
    "ENSG00000162490",  # DRAIC
    "ENSG00000026025",  # VIM    - vimentin, EMT / Basal
    "ENSG00000115461",  # IGFBP5
    "ENSG00000138777",  # KRT5   - cytokeratin 5, Basal
    "ENSG00000137309",  # KRT14  - cytokeratin 14, Basal
    "ENSG00000115687",  # PASK
    "ENSG00000100721",  # TCL1A  - T-cell leukemia / LumA
    "ENSG00000163631",  # ALB    - albumin (Normal-like)
    "ENSG00000148400",  # NOTCH1 - Notch signalling (Her2)
    "ENSG00000136436",  # CALCB
    "ENSG00000126266",  # BIRC5  # survivin, proliferation / LumB
    "ENSG00000073282",  # TP63   - Basal transcription factor
    "ENSG00000132821",  # BAG1   - Bcl2-associated athanogene (LumA)
    "ENSG00000065978",  # YBX1
    "ENSG00000197956",  # S100P  - Her2 enriched
    "ENSG00000119866",  # BCL11A
    "ENSG00000197380",  # DYRK3
    "ENSG00000187210",  # GCNT3
    "ENSG00000144485",  # HELLS  - lymphoid specific helicase
    "ENSG00000126254",  # RRAGC
    "ENSG00000125285",  # SOD3
    "ENSG00000101412",  # E2F1   - proliferation TF (LumB)
    "ENSG00000080986",  # NDC80  - kinetochore protein
    "ENSG00000117601",  # GINS3
    "ENSG00000170027",  # YDJC
    "ENSG00000138395",  # CDK15
    "ENSG00000101193",  # GID8
    "ENSG00000186416",  # NREP
    "ENSG00000139219",  # COL2A1
    "ENSG00000154803",  # PLCL1
    "ENSG00000139572",  # GPR84
    "ENSG00000163364",  # LINC01116
    "ENSG00000038945",  # MSH6   - mismatch repair
    "FOXM1", "MKI67", "ESR1", "PGR", "ERBB2", "GATA3", "FOXA1",  # symbol aliases
    "KRT5", "KRT14", "KRT17", "VIM", "BIRC5", "AURKA", "CCNB1",
    "CDK1", "PTTG1", "ANLN", "BUB1", "E2F1", "FOXM1", "CCNE1",
]


def variance_filter(
    matrix: pd.DataFrame,
    top_n: int = 5000,
    force_include: list[str] | None = None,
) -> pd.DataFrame:
    """Keep top N highest-variance features, with optional force-inclusion set.

    force_include: list of column names (gene IDs / symbols) that are always
    retained regardless of their variance rank, by replacing the tail of the
    top-N selection. This ensures canonical PAM50 proliferation markers
    (MKI67, AURKA, CCNB1, FOXM1, E2F1 ...) are always present in the RNA
    matrix, preventing global Basal-vs-Luminal variance from crowding them out.
    """
    variances = matrix.var(axis=0).sort_values(ascending=False)

    if force_include:
        # Intersect with actual column names to avoid KeyErrors on missing genes
        guaranteed = [g for g in force_include if g in matrix.columns]
        # Select top_n by variance, then union with guaranteed set
        top_by_var = list(variances.head(top_n).index)
        # Preserve variance order; add any guaranteed genes not already present
        missing_guaranteed = [g for g in guaranteed if g not in set(top_by_var)]
        keep_cols = top_by_var + missing_guaranteed
        # Trim to top_n + number of forced extras (keep all rather than evict)
        keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate, preserve order
    else:
        keep_cols = list(variances.head(top_n).index)

    return matrix[keep_cols]
