"""
BioNexus - Step 4: Finalize Labels + Platform Bias Check
---------------------------------------------------------
Re-filters patient_labels.csv down to the 733 patients that survived
preprocessing (shared_patient_index.csv), producing the labels file
you'll actually train against, and checks for platform bias.

Usage:
    python scripts/finalize_labels.py
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "processed" / "patient_labels.csv"
SHARED_INDEX_PATH = BASE_DIR / "data" / "processed" / "shared_patient_index.csv"

OUT_PATH = BASE_DIR / "data" / "processed" / "patient_labels_final.csv"


def main():
    labels = pd.read_csv(LABELS_PATH)
    shared = pd.read_csv(SHARED_INDEX_PATH)
    shared_ids = set(shared["patient_id"])

    print(f"Original labels file: {len(labels)} patients")
    print(f"Final shared cohort (post-preprocessing): {len(shared_ids)} patients\n")

    kept = labels[labels["patient_id"].isin(shared_ids)].copy()
    dropped = labels[~labels["patient_id"].isin(shared_ids)].copy()

    print(f"Kept:    {len(kept)} patients")
    print(f"Dropped: {len(dropped)} patients (should be ~302, the HM27-platform patients)\n")

    missing_labels = kept[kept["subtype"].isna() | (kept["subtype"].astype(str).str.strip() == "")]
    if len(missing_labels) > 0:
        print(f"[WARNING] {len(missing_labels)} of the kept 733 patients have NO subtype label:")
        print(missing_labels["patient_id"].to_string(index=False))
    else:
        print("All kept patients have a subtype label. No gaps.\n")

    kept_dist = kept["subtype"].value_counts(normalize=True).sort_index() * 100
    dropped_dist = dropped["subtype"].value_counts(normalize=True).sort_index() * 100

    comparison = pd.DataFrame({
        "kept_%": kept_dist,
        "dropped_%": dropped_dist,
    }).fillna(0).round(1)
    comparison["diff_pts"] = (comparison["kept_%"] - comparison["dropped_%"]).round(1)

    print("Subtype distribution: kept (733) vs dropped (302) patients")
    print(comparison.to_string())

    max_diff = comparison["diff_pts"].abs().max()
    print(f"\nLargest single-subtype gap: {max_diff:.1f} percentage points")
    if max_diff >= 10:
        print("[FLAG] This is a meaningful skew - the platform split is NOT random with "
              "respect to subtype.")
    else:
        print("Distributions look reasonably similar - dropping the HM27 patients doesn't "
              "appear to meaningfully bias your class balance.")

    kept.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(kept)} patients)")


if __name__ == "__main__":
    main()
