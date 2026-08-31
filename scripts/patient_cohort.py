"""
BioNexus - Patient ID Intersection Script (Step 1)
---------------------------------------------------
Reads the 3 GDC sample sheets (RNA-seq, Methylation, miRNA), maps each
downloaded file to its Case ID, and finds the patients present in ALL
THREE omics layers. Outputs a clean cohort list + a per-layer file
mapping table you can use for preprocessing.

Usage:
    python scripts/patient_cohort.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- BASE DIRECTORY (Project Root) ----
BASE_DIR = Path(__file__).resolve().parent.parent

# ---- SAMPLE SHEETS ----
RNA_SHEET = (
    BASE_DIR / "data" / "sample_sheets" / "RNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv"
)
MIRNA_SHEET = (
    BASE_DIR / "data" / "sample_sheets" / "MIRNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv"
)
METH_SHEET = (
    BASE_DIR
    / "data"
    / "sample_sheets"
    / "METHYLATION SAMPLE SHEET"
    / "gdc_sample_sheet.2026-08-23 (1).tsv"
)

# Fallback to root sample sheet paths if moved
if not RNA_SHEET.exists():
    RNA_SHEET = BASE_DIR / "RNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv"
if not MIRNA_SHEET.exists():
    MIRNA_SHEET = BASE_DIR / "MIRNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv"
if not METH_SHEET.exists():
    METH_SHEET = BASE_DIR / "METHYLATION SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23 (1).tsv"

RNA_DIR = BASE_DIR / "data" / "raw" / "rnaseq"
MIRNA_DIR = BASE_DIR / "data" / "raw" / "mirna"
METH_DIR = BASE_DIR / "data" / "raw" / "methylation"

OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sheet(path: Path, layer_name: str, file_keyword: str | None = None) -> pd.DataFrame:
    """Load a GDC sample sheet, keep only files actually present on disk."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    if file_keyword:
        df = df[df["File Name"].str.contains(file_keyword, na=False)]

    df = df[["Case ID", "Sample ID", "File Name", "Tissue Type", "Tumor Descriptor"]].copy()
    df = df.rename(
        columns={"File Name": f"{layer_name}_file", "Sample ID": f"{layer_name}_sample_id"}
    )
    df[f"{layer_name}_present"] = True

    n_total = len(df)
    print(f"[{layer_name}] {n_total} file entries loaded from sample sheet")
    return df


def main() -> None:
    rna = load_sheet(RNA_SHEET, "rna")
    mirna = load_sheet(MIRNA_SHEET, "mirna", file_keyword="mirnas.quantification")
    meth = load_sheet(METH_SHEET, "meth")

    def dedupe_to_primary(df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
        if "Tissue Type" in df.columns:
            preferred = df[df["Tissue Type"].str.contains("Tumor", case=False, na=False)]
            df = pd.concat([preferred, df]).drop_duplicates(subset="Case ID", keep="first")
        before = len(df)
        df = df.drop_duplicates(subset="Case ID", keep="first")
        print(f"[{layer_name}] {before} -> {len(df)} after de-duplicating to 1 sample/case")
        return df

    rna = dedupe_to_primary(rna, "rna")
    mirna = dedupe_to_primary(mirna, "mirna")
    meth = dedupe_to_primary(meth, "meth")

    def files_on_disk(directory: Path) -> set[str]:
        return {f.name for f in directory.rglob("*") if f.is_file() and not f.name.startswith(".")}

    rna_disk = files_on_disk(RNA_DIR)
    mirna_disk = files_on_disk(MIRNA_DIR)
    meth_disk = files_on_disk(METH_DIR)

    rna = rna[rna["rna_file"].isin(rna_disk)]
    mirna = mirna[mirna["mirna_file"].isin(mirna_disk)]
    meth = meth[meth["meth_file"].isin(meth_disk)]

    print(f"\n[rna]   {len(rna)} cases with file confirmed on disk")
    print(f"[mirna] {len(mirna)} cases with file confirmed on disk")
    print(f"[meth]  {len(meth)} cases with file confirmed on disk")

    merged = rna.merge(mirna, on="Case ID", how="inner", suffixes=("", "_mirna"))
    merged = merged.merge(meth, on="Case ID", how="inner", suffixes=("", "_meth"))

    print(f"\n=== FINAL COHORT: {len(merged)} patients with all 3 omics layers ===")

    out_path = OUT_DIR / "patient_cohort_mapping.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved full mapping table to: {out_path}")

    ids_path = OUT_DIR / "cohort_case_ids.txt"
    merged["Case ID"].drop_duplicates().to_csv(ids_path, index=False, header=False)
    print(f"Saved plain case ID list to: {ids_path}")


if __name__ == "__main__":
    main()
