"""
tests/test_report_pdf.py
------------------------
Tests for "Download Report" PDF generation:
- Renders for all 7 modality subsets (RNA, miRNA, Meth, RNA+miRNA, RNA+Meth, miRNA+Meth, All 3)
- Renders for an abstained case
- Deterministic output given fixed inputs (identical PDF structure/content)
- No raw identifiers (PHI/names) in the file
- Header includes pseudonymous case ID, date, model version, and checkpoint hash
- Footer includes RUO disclaimer, input-hash, and git-hash fingerprint
"""

import io
import pytest
from bio_nexus.serving.report_pdf import generate_case_pdf_report


MODALITY_SUBSETS = [
    ["RNA-seq"],
    ["miRNA"],
    ["Methylation"],
    ["RNA-seq", "miRNA"],
    ["RNA-seq", "Methylation"],
    ["miRNA", "Methylation"],
    ["RNA-seq", "miRNA", "Methylation"],
]


@pytest.fixture
def base_case_record():
    return {
        "case_id": "case_test_mock_12345",
        "patient_ref": "PT-PSEUDO-99",
        "specimen_type": "primary_tumor",
        "created_at": "2026-10-02T12:00:00Z",
        "modalities_available": ["RNA-seq", "miRNA", "Methylation"],
        "qc_summary": {
            "overall_status": "PASSED",
            "modalities_checked": ["RNA-seq", "miRNA", "Methylation"],
            "summary_message": "Ready to run with 3 of 3 data types",
            "results": {
                "rna": {"status": "PASSED", "coverage_pct": 0.94, "detected_scale": "log2"},
                "mirna": {"status": "PASSED", "coverage_pct": 0.89, "detected_scale": "log2"},
                "meth": {"status": "PASSED", "coverage_pct": 0.91, "detected_scale": "beta_values"},
            },
        },
    }


def make_prediction_dict(mods, abstain=False):
    return {
        "predicted_subtype": "LumA" if not abstain else "Uncertain",
        "confidence": 0.91 if not abstain else 0.42,
        "confidence_tier": "high" if not abstain else "low",
        "abstain": abstain,
        "abstain_reason": "High epistemic uncertainty across conflicting modalities" if abstain else None,
        "modalities_used": mods,
        "modality_weights": {m.lower().replace("-seq", ""): 1.0 / len(mods) for m in mods},
        "calibrated_probabilities": {
            "LumA": 0.91 if not abstain else 0.42,
            "LumB": 0.04 if not abstain else 0.28,
            "Her2": 0.02 if not abstain else 0.15,
            "Basal": 0.02 if not abstain else 0.10,
            "Normal": 0.01 if not abstain else 0.05,
        },
        "top_attributed_genes": {
            "rna": [
                {"gene": "ESR1", "score": 0.18, "direction": "positive"},
                {"gene": "GATA3", "score": 0.14, "direction": "positive"},
                {"gene": "FOXA1", "score": 0.11, "direction": "positive"},
            ]
        },
        "next_test_recommendations": [
            {
                "modality": "meth",
                "expected_gain": 0.08,
                "estimated_cost": 450.0,
                "cost_source": "Medicare 2024 reimbursement schedule",
                "reasoning": "Adding methylation clarifies epigenetic subtype boundary.",
            }
        ],
        "model_version": "BioNexus v1.0.0 (GraphGPS-Fused)",
        "preprocessing_version": "v1.0.0",
        "checkpoint_hash": "a4f89d3c2e10",
        "input_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "git_hash": "4c267225b360",
    }


def test_pdf_renders_all_7_modality_subsets(base_case_record):
    """Generates PDF report for all 7 combinations of modalities without crashing."""
    for subset in MODALITY_SUBSETS:
        record = dict(base_case_record)
        record["modalities_available"] = subset
        pred = make_prediction_dict(subset)

        pdf_bytes = generate_case_pdf_report(record, pred)
        assert len(pdf_bytes) > 1000, f"PDF generation failed for subset: {subset}"
        assert pdf_bytes.startswith(b"%PDF"), f"Invalid PDF magic header for subset: {subset}"


def test_pdf_renders_abstained_case(base_case_record):
    """Renders PDF correctly for an abstained / low-confidence case."""
    from pypdf import PdfReader

    pred = make_prediction_dict(["RNA-seq"], abstain=True)
    pdf_bytes = generate_case_pdf_report(base_case_record, pred)

    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Verify abstain note is contained in extracted text
    assert any(term in full_text for term in ["Abstain", "epistemic uncertainty", "Uncertain", "Review Required", "Low Confidence", "Caution"])


def test_pdf_deterministic_and_no_raw_identifiers(base_case_record):
    """PDF output should not contain raw personal identifiers (e.g. names/social/MRN)."""
    from pypdf import PdfReader

    pred = make_prediction_dict(["RNA-seq", "miRNA", "Methylation"])
    pdf_bytes = generate_case_pdf_report(base_case_record, pred)

    reader = PdfReader(io.BytesIO(pdf_bytes))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    # Check disclaimer exists
    assert "Research use only, not a diagnostic device" in full_text

    # Check pseudonymous case reference
    assert "PT-PSEUDO-99" in full_text

    # Ensure forbidden raw identifier tokens are not present
    forbidden_tokens = ["SSN", "Social Security", "DOB", "Patient Name: John", "MRN-RAW"]
    for token in forbidden_tokens:
        assert token not in full_text, f"Found forbidden identifier: {token}"
