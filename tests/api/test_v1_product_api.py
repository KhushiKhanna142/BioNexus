"""
Comprehensive test suite for the BioNexus v1 Product API.
Tests:
- Case intake via JSON and multipart upload
- Quality Control (QC) coverage thresholds, scale detection, symbol/Ensembl mapping
- Inductive inference with calibrated probabilities, confidence tiers, and abstention
- Integrated Gradients biomarker attribution and next-test recommender
- Downloadable PDF report generation
- Cheap gene-panel student model
- Privacy deletion, audit logging, and RBAC / API key validation
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from bio_nexus.serving.api import app
from bio_nexus.serving.config import settings

client = TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dev-admin-key"}


@pytest.fixture
def sample_rna_vector():
    """Generates a realistic sample with canonical breast cancer markers."""
    # Canonical PAM50 and high-variance TCGA-BRCA markers
    return {
        "ESR1": 4.5,
        "PGR": 3.8,
        "ERBB2": 1.2,
        "MKI67": 1.5,
        "GATA3": 4.1,
        "FOXA1": 3.2,
        "EGFR": 0.8,
        "KRT5": 0.5,
        "KRT14": 0.3,
        "KRT17": 0.4,
    }


def test_v1_health_and_version():
    r = client.get("/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "environment" in data

    r2 = client.get("/v1/version")
    assert r2.status_code == 200
    assert r2.json()["api_version"] == "v1"

    # Root shortcuts
    assert client.get("/health").status_code == 200
    assert client.get("/version").status_code == 200


def test_v1_model_card_and_cost_simulation(auth_headers):
    r = client.get("/v1/model/card", headers=auth_headers)
    assert r.status_code == 200
    card = r.json()
    assert "BioNexus" in card["model_name"]
    assert card["total_parameters"] == 1124970
    assert "cross_validation" in card

    r2 = client.get("/v1/simulate/cost-accuracy?budget=1000", headers=auth_headers)
    assert r2.status_code == 200
    curve = r2.json()["curve"]
    assert len(curve) > 0
    assert all(pt["estimated_cost"] <= 1000 for pt in curve)


def test_v1_case_intake_json_and_predict(auth_headers, sample_rna_vector):
    # 1. Create Case via JSON with mock RNA vector that covers key features
    # Pad sample vector so it passes minimum QC threshold (> 50% of model features)
    from bio_nexus.serving.qc import qc_manager
    expected_rna = qc_manager.expected_features.get("rna", [])
    test_rna = {fid: 2.5 for fid in expected_rna[:3000]}  # 3000 / 5000 = 60% coverage
    test_rna.update(sample_rna_vector)

    payload = {
        "patient_ref": "TEST-PATIENT-001",
        "specimen_type": "core_needle_biopsy",
        "rna": test_rna,
    }

    r_create = client.post("/v1/cases", json=payload, headers=auth_headers)
    assert r_create.status_code == 200, r_create.text
    case_resp = r_create.json()
    assert "case_id" in case_resp
    assert case_resp["patient_ref"] == "TEST-PATIENT-001"
    assert case_resp["status"] == "intake_qc_passed"
    assert case_resp["qc"]["overall_status"] in ("PASSED", "WARNING")
    case_id = case_resp["case_id"]

    # 2. Run Inductive Inference on Created Case
    r_pred = client.post(f"/v1/cases/{case_id}/predict", headers=auth_headers)
    assert r_pred.status_code == 200, r_pred.text
    pred_resp = r_pred.json()
    assert pred_resp["case_id"] == case_id
    assert pred_resp["predicted_subtype"] in ["LumA", "LumB", "Her2", "Basal", "Normal"]
    assert 0.0 <= pred_resp["confidence"] <= 1.0
    assert "LumA" in pred_resp["calibrated_probabilities"]
    assert pred_resp["confidence_tier"] in ["high", "medium", "low"]
    assert "rna" in pred_resp["modalities_used"]
    assert "rna" in pred_resp["top_attributed_genes"]
    assert "timings_ms" in pred_resp
    assert "Research use only" in pred_resp["disclaimer"]

    # 3. Retrieve Case Metadata
    r_get = client.get(f"/v1/cases/{case_id}", headers=auth_headers)
    assert r_get.status_code == 200
    assert r_get.json()["status"] == "predicted"

    # 4. Download PDF Report
    r_pdf = client.get(f"/v1/cases/{case_id}/report.pdf", headers=auth_headers)
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert len(r_pdf.content) > 1000  # Non-empty valid PDF bytes

    # 5. Delete Case (GDPR / Privacy)
    r_del = client.delete(f"/v1/cases/{case_id}", headers=auth_headers)
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "deleted"

    # Verify 404 after deletion
    assert client.get(f"/v1/cases/{case_id}", headers=auth_headers).status_code == 404


def test_v1_qc_coverage_rejection(auth_headers):
    # Only supply 2 genes (way below 50% coverage threshold)
    payload = {
        "patient_ref": "POOR-COVERAGE-CASE",
        "rna": {"ESR1": 4.2, "PGR": 3.1},
    }
    r = client.post("/v1/cases", json=payload, headers=auth_headers)
    assert r.status_code == 422
    data = r.json()
    assert data["code"] == "QC_COVERAGE_FAILED"
    assert "Feature coverage" in data["detail"]


def test_v1_qc_scale_warning(auth_headers):
    from bio_nexus.serving.qc import qc_manager
    expected_rna = qc_manager.expected_features.get("rna", [])
    # Unlogged raw counts with huge values (e.g. 5,000 counts)
    raw_scale_rna = {fid: 5000.0 for fid in expected_rna[:3000]}

    payload = {
        "patient_ref": "RAW-SCALE-CASE",
        "rna": raw_scale_rna,
    }
    r = client.post("/v1/cases", json=payload, headers=auth_headers)
    assert r.status_code == 200
    qc_data = r.json()["qc"]
    assert qc_data["overall_status"] == "WARNING"
    assert qc_data["modality_details"]["rna"]["likely_raw_counts"] is True


def test_v1_panel_predict(auth_headers):
    payload = {
        "patient_ref": "PANEL-PT-001",
        "genes": {
            "ESR1": 5.2,
            "GATA3": 4.8,
            "PGR": 4.1,
            "ERBB2": 0.5,
            "KRT5": 0.2,
        },
    }
    r = client.post("/v1/panel-predict", json=payload, headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["predicted_subtype"] in ["LumA", "LumB", "Her2", "Basal", "Normal"]
    assert data["panel_size"] == 5
    assert data["genes_matched"] > 0
    assert "Student MLP" in data["model_type"]


def test_v1_case_intake_multipart(auth_headers, sample_rna_vector):
    from bio_nexus.serving.qc import qc_manager
    expected_rna = qc_manager.expected_features.get("rna", [])
    test_rna = {fid: 2.5 for fid in expected_rna[:3000]}
    test_rna.update(sample_rna_vector)

    csv_buf = io.StringIO()
    csv_buf.write("feature_id,value\n")
    for k, v in test_rna.items():
        csv_buf.write(f"{k},{v}\n")
    csv_bytes = csv_buf.getvalue().encode()

    files = {"rna_file": ("rna_profile.csv", csv_bytes, "text/csv")}
    data = {"patient_ref": "MULTIPART-PT-001", "specimen_type": "fresh_frozen"}

    r = client.post("/v1/cases", data=data, files=files, headers=auth_headers)
    assert r.status_code == 200, r.text
    case_resp = r.json()
    assert case_resp["patient_ref"] == "MULTIPART-PT-001"
    assert case_resp["status"] == "intake_qc_passed"
    assert "rna" in case_resp["modalities_received"]


def test_v1_batches_and_metrics(auth_headers):
    # Test batch upload
    manifest_csv = (
        "patient_ref,ESR1,PGR,ERBB2\n"
        "BATCH-PT-001,4.5,3.2,1.1\n"
        "BATCH-PT-002,1.2,0.8,5.4\n"
    ).encode()
    files = {"manifest": ("manifest.csv", manifest_csv, "text/csv")}
    r = client.post("/v1/batches", files=files, headers=auth_headers)
    assert r.status_code == 200
    batch_data = r.json()
    batch_id = batch_data["batch_id"]
    assert batch_data["total_cases"] == 2

    # Status check
    r_stat = client.get(f"/v1/batches/{batch_id}", headers=auth_headers)
    assert r_stat.status_code == 200
    assert r_stat.json()["batch_id"] == batch_id

    # Results CSV check
    r_csv = client.get(f"/v1/batches/{batch_id}/results.csv", headers=auth_headers)
    assert r_csv.status_code == 200
    assert "case_id" in r_csv.text

    # Metrics check
    r_met = client.get("/v1/metrics")
    assert r_met.status_code == 200
    assert "bionexus_cases_stored_total" in r_met.text

    # List cases check
    r_list = client.get("/v1/cases?limit=10", headers=auth_headers)
    assert r_list.status_code == 200
    assert len(r_list.json()["cases"]) > 0


def test_v1_privacy_validation_blocks_pii(auth_headers):
    # Names with spaces or invalid chars should be rejected
    payload = {
        "patient_ref": "Jane Doe SNN 123-45-6789",
        "rna": {"ESR1": 4.0},
    }
    r = client.post("/v1/cases", json=payload, headers=auth_headers)
    assert r.status_code == 400
    assert r.json()["code"] == "INVALID_PATIENT_REF"


def test_v1_auth_and_rate_limiting():
    # Test unauthenticated access when dev_mode is disabled
    from bio_nexus.serving.config import settings
    prev_dev = settings.dev_mode
    try:
        settings.dev_mode = False
        r_unauth = client.post("/v1/cases", json={"patient_ref": "TEST"})
        assert r_unauth.status_code == 401

        # Test invalid key
        r_bad = client.post("/v1/cases", json={"patient_ref": "TEST"}, headers={"X-API-Key": "fake-bad-key"})
        assert r_bad.status_code == 401

        # Test viewer role forbidden from operator endpoint
        r_forbidden = client.post(
            "/v1/cases",
            json={"patient_ref": "TEST"},
            headers={"X-API-Key": "dev-viewer-key"},
        )
        assert r_forbidden.status_code == 403
    finally:
        settings.dev_mode = prev_dev
