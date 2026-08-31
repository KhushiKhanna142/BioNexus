"""
BioNexus - Step 3: Patient Label Extraction Script
---------------------------------------------------
Reads the cBioPortal PAM50 clinical data export, extracts Patient ID +
Subtype, strips the "BRCA_" prefix, and filters down to only the
patients in your final cohort (cohort_case_ids.txt). Flags any cohort
patients missing a subtype label.

Usage:
    python scripts/extract_labels.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

CLINICAL_TSV = BASE_DIR / "data" / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv"
COHORT_IDS_PATH = BASE_DIR / "data" / "processed" / "cohort_case_ids.txt"

OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "patient_labels.csv"


def main():
    clinical = pd.read_csv(CLINICAL_TSV, sep="\t")
    clinical.columns = [c.strip() for c in clinical.columns]

    if "Patient ID" not in clinical.columns or "Subtype" not in clinical.columns:
        raise ValueError(
            f"Expected 'Patient ID' and 'Subtype' columns. Found: {list(clinical.columns)}"
        )

    labels = clinical[["Patient ID", "Subtype"]].copy()
    labels = labels.rename(columns={"Patient ID": "patient_id", "Subtype": "subtype"})

    before = len(labels)
    labels = labels.drop_duplicates(subset="patient_id", keep="first")
    print(f"[labels] {before} -> {len(labels)} rows after de-duplicating to 1 row/patient")

    labels["subtype"] = labels["subtype"].str.replace("BRCA_", "", regex=False)

    cohort_ids = pd.read_csv(COHORT_IDS_PATH, header=None, names=["patient_id"])
    print(f"[cohort] {len(cohort_ids)} patient IDs loaded from cohort_case_ids.txt")

    merged = cohort_ids.merge(labels, on="patient_id", how="left")

    missing = merged[merged["subtype"].isna() | (merged["subtype"].str.strip() == "")]
    if len(missing) > 0:
        print(f"\n[WARNING] {len(missing)} cohort patients have NO subtype label:")
        print(missing["patient_id"].to_string(index=False))
    else:
        print("\nAll cohort patients have a subtype label. No gaps.")

    print("\nSubtype distribution:")
    print(merged["subtype"].value_counts(dropna=False).to_string())

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
