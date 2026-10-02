"""
scripts/generate_sample_reports.py
----------------------------------
Generates three realistic clinical sample reports in docs/sample_reports/:
1. High confidence case (multi-omics, calibrated confidence > 0.85, auto-report)
2. Low confidence case (ambiguous subtype call, low confidence, triggers next-test recommendation)
3. Single modality case (RNA-seq alone, partial panel)
"""

from pathlib import Path
from bio_nexus.serving.report_pdf import generate_case_pdf_report


def main():
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "docs" / "sample_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample reports in: {out_dir}")

    # 1. High Confidence Sample
    case_high = {
        "case_id": "CASE-BRCA-9201",
        "patient_ref": "PATIENT-DEMO-001",
        "created_at": "2026-10-02T18:30:00Z",
        "specimen_type": "primary_tumor",
        "modalities": ["rna", "mirna", "meth"],
        "input_hash": "sha256-a1b2c3d4e5f6789012345678abcdef01",
        "git_hash": "4c267225b360",
        "qc_summary": {
            "overall_status": "PASSED",
            "results": {
                "rna": {"status": "PASSED", "mapped_count": 4890, "total_expected": 5000, "coverage_pct": 0.978, "likely_raw_counts": False},
                "mirna": {"status": "PASSED", "mapped_count": 1820, "total_expected": 1881, "coverage_pct": 0.968, "likely_raw_counts": False},
                "meth": {"status": "PASSED", "mapped_count": 3810, "total_expected": 3957, "coverage_pct": 0.963, "likely_raw_counts": False},
            },
        },
    }
    pred_high = {
        "predicted_subtype": "Basal",
        "confidence": 0.942,
        "confidence_tier": "HIGH_CONFIDENCE",
        "triage_status": "AUTO_REPORT",
        "triage_reason": "High calibrated confidence (94.2% >= 80.0%) with Concordant Multi-Omics Signatures.",
        "abstain": False,
        "probabilities": {"Basal": 0.942, "Her2": 0.031, "LumB": 0.015, "Normal": 0.008, "LumA": 0.004},
        "modality_weights": {"rna": 0.54, "mirna": 0.28, "meth": 0.18},
        "top_biomarkers": [
            {"gene": "FOXC1", "symbol": "FOXC1", "feature_id": "ENSG00000054598", "layer": "rna", "importance": 0.284, "direction": "+", "clinical_significance": "Established basal-like diagnostic transcription factor."},
            {"gene": "SOX10", "symbol": "SOX10", "feature_id": "ENSG00000100146", "layer": "rna", "importance": 0.241, "direction": "+", "clinical_significance": "Neural crest stem marker enriched in triple-negative Basal tumors."},
            {"gene": "ESR1", "symbol": "ESR1", "feature_id": "ENSG00000091831", "layer": "rna", "importance": -0.215, "direction": "-", "clinical_significance": "Estrogen receptor downregulation confirms non-luminal phenotype."},
        ],
        "next_test_recommendations": [],
    }
    pdf_high = generate_case_pdf_report(case_high, pred_high)
    path_high = out_dir / "sample_high_confidence.pdf"
    with open(path_high, "wb") as f:
        f.write(pdf_high)
    print(f"Wrote: {path_high}")

    # 2. Low Confidence Sample (Triggers Clinician Review + Next-Test Recommendation)
    case_low = {
        "case_id": "CASE-BRCA-4412",
        "patient_ref": "PATIENT-DEMO-002",
        "created_at": "2026-10-02T19:15:00Z",
        "specimen_type": "primary_tumor",
        "modalities": ["rna"],
        "input_hash": "sha256-f9e8d7c6b5a4321098765432fedcba98",
        "git_hash": "4c267225b360",
        "qc_summary": {
            "overall_status": "WARNING",
            "results": {
                "rna": {"status": "WARNING", "mapped_count": 3950, "total_expected": 5000, "coverage_pct": 0.790, "likely_raw_counts": False},
            },
        },
    }
    pred_low = {
        "predicted_subtype": "LumB",
        "confidence": 0.548,
        "confidence_tier": "LOW_CONFIDENCE",
        "triage_status": "NEEDS_MORE_DATA",
        "triage_reason": "Low confidence (54.8% < 60.0%) with LumA vs LumB boundary ambiguity.",
        "abstain": True,
        "probabilities": {"LumB": 0.548, "LumA": 0.382, "Her2": 0.045, "Normal": 0.015, "Basal": 0.010},
        "modality_weights": {"rna": 1.0, "mirna": 0.0, "meth": 0.0},
        "top_biomarkers": [
            {"gene": "MKI67", "symbol": "MKI67", "feature_id": "ENSG00000148773", "layer": "rna", "importance": 0.165, "direction": "+", "clinical_significance": "Proliferation marker driving LumB distinction."},
            {"gene": "ESR1", "symbol": "ESR1", "feature_id": "ENSG00000091831", "layer": "rna", "importance": 0.142, "direction": "+", "clinical_significance": "Luminal receptor expressed in both LumA and LumB."},
        ],
        "next_test_recommendations": [
            {"assay": "miRNA Sequencing", "expected_gain": 0.082, "cost": 350.0, "turnaround": "1-2 days", "rationale": "Adding miRNA profiling provides highest subtype differentiation power to resolve LumA vs LumB boundary."},
            {"assay": "DNA Methylation", "expected_gain": 0.045, "cost": 450.0, "turnaround": "5-7 days", "rationale": "Methylation status at ESR1/PGR promoter loci clarifies epigenetic silencing."},
        ],
    }
    pdf_low = generate_case_pdf_report(case_low, pred_low)
    path_low = out_dir / "sample_low_confidence.pdf"
    with open(path_low, "wb") as f:
        f.write(pdf_low)
    print(f"Wrote: {path_low}")

    # 3. Single Modality Sample (RNA-only, Clinician Review)
    case_single = {
        "case_id": "CASE-BRCA-7703",
        "patient_ref": "PATIENT-DEMO-003",
        "created_at": "2026-10-02T20:00:00Z",
        "specimen_type": "primary_tumor",
        "modalities": ["rna"],
        "input_hash": "sha256-11223344556677889900aabbccddeeff",
        "git_hash": "4c267225b360",
        "qc_summary": {
            "overall_status": "PASSED",
            "results": {
                "rna": {"status": "PASSED", "mapped_count": 4720, "total_expected": 5000, "coverage_pct": 0.944, "likely_raw_counts": False},
            },
        },
    }
    pred_single = {
        "predicted_subtype": "Her2",
        "confidence": 0.725,
        "confidence_tier": "MEDIUM_CONFIDENCE",
        "triage_status": "CLINICIAN_REVIEW",
        "triage_reason": "Moderate confidence (72.5%) based on single-modality RNA transcriptomics.",
        "abstain": False,
        "probabilities": {"Her2": 0.725, "LumB": 0.160, "Basal": 0.065, "LumA": 0.035, "Normal": 0.015},
        "modality_weights": {"rna": 1.0, "mirna": 0.0, "meth": 0.0},
        "top_biomarkers": [
            {"gene": "ERBB2", "symbol": "ERBB2", "feature_id": "ENSG00000141736", "layer": "rna", "importance": 0.312, "direction": "+", "clinical_significance": "HER2 receptor tyrosine kinase amplification hallmark."},
            {"gene": "GRB7", "symbol": "GRB7", "feature_id": "ENSG00000141738", "layer": "rna", "importance": 0.228, "direction": "+", "clinical_significance": "Co-amplified in ERBB2 amplicon core."},
        ],
        "next_test_recommendations": [
            {"assay": "miRNA Sequencing", "expected_gain": 0.058, "cost": 350.0, "turnaround": "1-2 days", "rationale": "Secondary miRNA panel can boost confidence into auto-report tier."},
        ],
    }
    pdf_single = generate_case_pdf_report(case_single, pred_single)
    path_single = out_dir / "sample_single_modality.pdf"
    with open(path_single, "wb") as f:
        f.write(pdf_single)
    print(f"Wrote: {path_single}")

    print("Success! Generated all 3 clinical sample reports.")


if __name__ == "__main__":
    main()
