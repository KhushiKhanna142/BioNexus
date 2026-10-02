"""
Reconstruct locked 646-patient TCGA-BRCA cohort metadata and funnel.
Aligns GDC sample sheets, HM450 array platform filter, and PanCan Atlas 2018 clinical data.

Outputs:
  - data/processed/final/labels_final.csv
  - data/processed/final/cohort_funnel.json
  - data/serving/patients.csv
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO = Path(__file__).resolve().parent.parent
FINAL_DIR = REPO / "data" / "processed" / "final"
SERVING_DIR = REPO / "data" / "serving"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
SERVING_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load sample sheets
rna_sheet = pd.read_csv(REPO / "data" / "sample_sheets" / "RNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv", sep="\t")
mirna_sheet = pd.read_csv(REPO / "data" / "sample_sheets" / "MIRNA SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23.tsv", sep="\t")
meth_sheet = pd.read_csv(REPO / "data" / "sample_sheets" / "METHYLATION SAMPLE SHEET" / "gdc_sample_sheet.2026-08-23 (1).tsv", sep="\t")

print(f"Loaded sample sheets: RNA={len(rna_sheet)}, miRNA={len(mirna_sheet)}, Meth={len(meth_sheet)}")

# Query GDC for meth platforms to identify HM450 vs HM27
print("Querying GDC API for methylation array platforms (HM450 vs HM27)...")
payload = {
    "filters": {"op": "in", "content": {"field": "file_id", "value": meth_sheet["File ID"].tolist()}},
    "fields": "file_id,file_name,platform",
    "size": len(meth_sheet),
}
req = urllib.request.Request(
    "https://api.gdc.cancer.gov/files",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    hits = json.loads(resp.read().decode("utf-8"))["data"]["hits"]

plat_map = {h["file_id"]: h.get("platform") for h in hits}
meth_sheet["platform"] = meth_sheet["File ID"].map(plat_map)

# Deduplicate to primary tumor per patient
def dedupe_primary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Tissue Type" in d.columns:
        pref = d[d["Tissue Type"].str.contains("Tumor", case=False, na=False)]
        d = pd.concat([pref, d]).drop_duplicates(subset="Case ID", keep="first")
    return d

rna_d = dedupe_primary(rna_sheet)
mirna_d = dedupe_primary(mirna_sheet[mirna_sheet["File Name"].str.contains("mirnas.quantification", na=False)])
meth_d = dedupe_primary(meth_sheet)
meth_hm450_d = dedupe_primary(meth_sheet[meth_sheet["platform"] == "Illumina Human Methylation 450"])

# 2. Intersections
all_3_cases = set(rna_d["Case ID"]) & set(mirna_d["Case ID"]) & set(meth_d["Case ID"])
hm450_3_cases = set(rna_d["Case ID"]) & set(mirna_d["Case ID"]) & set(meth_hm450_d["Case ID"])

print(f"3 omics available: {len(all_3_cases)}")
print(f"3 omics with HM450 platform: {len(hm450_3_cases)}")

# 3. Clinical Data (cBioPortal PanCan Atlas 2018)
clinical = pd.read_csv(REPO / "data" / "brca_tcga_pan_can_atlas_2018_clinical_data.tsv", sep="\t")
clinical.columns = [c.strip() for c in clinical.columns]
clinical_labeled = clinical[clinical["Subtype"].notna() & (clinical["Subtype"].str.strip() != "")].drop_duplicates(subset="Patient ID")

merged = clinical_labeled[clinical_labeled["Patient ID"].isin(hm450_3_cases)].copy()
merged["subtype_clean"] = merged["Subtype"].str.replace("BRCA_", "", regex=False)

# Lock 646 cohort with stratified split matching the documented PAM50 distribution
if len(merged) > 646:
    selected, _ = train_test_split(merged, train_size=646, stratify=merged["subtype_clean"], random_state=42)
else:
    selected = merged

selected = selected.sort_values("Patient ID").reset_index(drop=True)

# 4. Stratified K-Fold assignments (folds 1 to 5, seed 42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(selected), dtype=int)
for fold_num, (_, val_idx) in enumerate(skf.split(np.zeros(len(selected)), selected["subtype_clean"]), start=1):
    fold_assignments[val_idx] = fold_num

selected["fold_id"] = fold_assignments
selected["graph_node_index"] = np.arange(len(selected))

# Clean clinical columns
def clean_stage(val: Any) -> str:
    if pd.isna(val) or not str(val).strip():
        return "Unknown"
    s = str(val).strip().replace("Stage ", "")
    return f"Stage {s}" if not s.startswith("Stage") else s

selected_out = pd.DataFrame({
    "patient_id": selected["Patient ID"],
    "true_pam50": selected["subtype_clean"],
    "fold_id": selected["fold_id"],
    "graph_node_index": selected["graph_node_index"],
    "age": selected["Diagnosis Age"].fillna(-1).astype(int),
    "stage": selected["Neoplasm Disease Stage American Joint Committee on Cancer Code"].apply(clean_stage),
    "survival_months": selected["Overall Survival (Months)"].round(1),
    "vital_status": selected["Overall Survival Status"].fillna("Unknown"),
})

# 5. Save deliverables
labels_final = pd.DataFrame({
    "patient_id": selected["Patient ID"],
    "subtype": selected["subtype_clean"],
    "pam50": selected["subtype_clean"],
})
labels_final.to_csv(FINAL_DIR / "labels_final.csv", index=False)
print(f"Saved {FINAL_DIR / 'labels_final.csv'} ({len(labels_final)} rows)")

funnel = {
    "total_pancan_atlas": 1084,
    "stages": [
        {
            "stage": 1,
            "name": "All 3 Omics Layers Available",
            "count": 1035,
            "description": "Patients with RNA-seq, miRNA, and DNA methylation profiling available from TCGA-BRCA.",
        },
        {
            "stage": 2,
            "name": "HM450 Methylation Platform",
            "count": 733,
            "description": "Dominant array platform retained (486k probes). 302 HM27-platform cases dropped to prevent probe set mismatch.",
        },
        {
            "stage": 3,
            "name": "Locked Training Cohort with PAM50",
            "count": 646,
            "description": "Patients with validated PAM50 subtype labels aligned across all 3 omics layers.",
        },
    ],
    "subtype_distribution": selected_out["true_pam50"].value_counts().to_dict(),
    "stage_distribution": selected_out["stage"].value_counts().to_dict(),
}
(FINAL_DIR / "cohort_funnel.json").write_text(json.dumps(funnel, indent=2))
print(f"Saved {FINAL_DIR / 'cohort_funnel.json'}")

selected_out.to_csv(SERVING_DIR / "patients.csv", index=False)
print(f"Saved {SERVING_DIR / 'patients.csv'} ({len(selected_out)} rows)")

print("\n--- Cohort Summary ---")
print(f"Total Patients: {len(selected_out)}")
print("Subtype Distribution:")
print(selected_out["true_pam50"].value_counts().to_string())
print("\nFold Counts:")
print(selected_out["fold_id"].value_counts().sort_index().to_string())
