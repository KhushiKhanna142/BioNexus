"""
BioNexus - Step 5: Final Cohort Lock-in
----------------------------------------
Filters the 733-patient omics matrices + labels down to the 646
patients that have ALL THREE omics layers AND a subtype label. This is
the final, ready-to-train cohort - saves aligned matrices + labels so
every downstream script (graph construction, model training) reads
from one consistent source.

Usage:
    python scripts/final_cohort.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

RNA_PATH = PROC_DIR / "rna_matrix.csv"
MIRNA_PATH = PROC_DIR / "mirna_matrix.csv"
METH_PATH = PROC_DIR / "meth_matrix.csv"
LABELS_PATH = PROC_DIR / "patient_labels_final.csv"

OUT_DIR = PROC_DIR / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    labels = pd.read_csv(LABELS_PATH)
    labeled = labels[labels["subtype"].notna() & (labels["subtype"].astype(str).str.strip() != "")]
    print(f"Patients with omics + label file entry: {len(labels)}")
    print(f"Patients with a non-missing subtype label: {len(labeled)}")

    final_ids = set(labeled["patient_id"])

    rna = pd.read_csv(RNA_PATH, index_col=0)
    mirna = pd.read_csv(MIRNA_PATH, index_col=0)
    meth = pd.read_csv(METH_PATH, index_col=0)

    missing_in_omics = final_ids - (set(rna.index) & set(mirna.index) & set(meth.index))
    if missing_in_omics:
        print(f"[WARNING] {len(missing_in_omics)} labeled patients missing from omics matrices - dropping them too")
        final_ids -= missing_in_omics

    final_ids = sorted(final_ids)
    print(f"\nFinal locked-in cohort: {len(final_ids)} patients")

    rna_final = rna.loc[final_ids]
    mirna_final = mirna.loc[final_ids]
    meth_final = meth.loc[final_ids]
    labels_final = labeled[labeled["patient_id"].isin(final_ids)].set_index("patient_id").loc[final_ids].reset_index()

    rna_final.to_csv(OUT_DIR / "rna_matrix_final.csv")
    mirna_final.to_csv(OUT_DIR / "mirna_matrix_final.csv")
    meth_final.to_csv(OUT_DIR / "meth_matrix_final.csv")
    labels_final.to_csv(OUT_DIR / "labels_final.csv", index=False)
    pd.Series(final_ids, name="patient_id").to_csv(OUT_DIR / "final_patient_index.csv", index=False)

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  rna_matrix_final.csv   ({rna_final.shape[0]} x {rna_final.shape[1]})")
    print(f"  mirna_matrix_final.csv ({mirna_final.shape[0]} x {mirna_final.shape[1]})")
    print(f"  meth_matrix_final.csv  ({meth_final.shape[0]} x {meth_final.shape[1]})")
    print(f"  labels_final.csv       ({len(labels_final)} patients)")
    print(f"\nSubtype distribution in final cohort:")
    print(labels_final["subtype"].value_counts().to_string())


if __name__ == "__main__":
    main()
