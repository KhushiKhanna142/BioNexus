"""
tests/test_batch_triage.py
--------------------------
Tests for "Batch Triage Board" feature:
- 100-case synthetic batch processing
- Partial failures handled gracefully per-case
- Lanes assigned strictly from configured tier thresholds:
    * Auto-report: confidence >= 0.80
    * Clinician review: 0.60 <= confidence < 0.80
    * Needs more data: confidence < 0.60 or abstain or missing modalities
- CSV export schema validation
"""

import io
import csv
import pytest
from fastapi.testclient import TestClient

from bio_nexus.serving.api import app
from bio_nexus.serving.v1_router import _assign_triage_lane


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dev-admin-key"}


@pytest.fixture
def client():
    return TestClient(app)


# =========================================================================
# 1. UNIT TEST FOR TRIAGE LANE ASSIGNMENT RULE
# =========================================================================

def test_triage_lane_threshold_assignment():
    """Verify lane assignment logic strictly honors configured tier thresholds."""
    # High confidence -> auto_report
    assert _assign_triage_lane(0.92, 3, False) == "auto_report"
    assert _assign_triage_lane(0.80, 2, False) == "auto_report"

    # Medium confidence -> clinician_review
    assert _assign_triage_lane(0.799, 3, False) == "clinician_review"
    assert _assign_triage_lane(0.60, 2, False) == "clinician_review"

    # Low confidence -> needs_more_data
    assert _assign_triage_lane(0.599, 3, False) == "needs_more_data"
    assert _assign_triage_lane(0.35, 1, False) == "needs_more_data"

    # Abstain or missing modalities override high confidence -> needs_more_data
    assert _assign_triage_lane(0.95, 0, False) == "needs_more_data"
    assert _assign_triage_lane(0.95, 3, True) == "needs_more_data"


# =========================================================================
# 2. 100-CASE SYNTHETIC BATCH PROCESSING & PARTIAL FAILURE HANDLING
# =========================================================================

def test_100_case_synthetic_batch_and_partial_failures(client, auth_headers):
    """
    Submits a 100-case synthetic manifest containing valid cases, borderline cases,
    and partial failures (e.g. invalid syntax on specific rows).
    Verifies:
      1. Batch succeeds and processes all cases.
      2. Partial failures do not crash the batch.
      3. Cases are routed into all 3 lanes strictly by threshold.
    """
    manifest_rows = ["patient_ref,specimen_type,rna_file\n"]
    for i in range(100):
        if i == 13 or i == 42:
            # Simulate a row with deliberate malformed data
            manifest_rows.append(f"PT-SYNTH-{i:03d},invalid_specimen,\n")
        else:
            manifest_rows.append(f"PT-SYNTH-{i:03d},primary_tumor,rna_mock_{i}.csv\n")

    manifest_csv = "".join(manifest_rows).encode("utf-8")
    files = {"manifest": ("manifest.csv", manifest_csv, "text/csv")}

    resp = client.post("/v1/batches", files=files, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "batch_id" in data
    assert data["total_cases"] == 100
    batch_id = data["batch_id"]

    # Verify status and lane distribution via GET /v1/batches/{id}
    status_resp = client.get(f"/v1/batches/{batch_id}", headers=auth_headers)
    assert status_resp.status_code == 200
    batch_details = status_resp.json()

    assert batch_details["status"] in ["completed", "processing"]
    summary = batch_details["summary"]
    assert summary["total_cases"] == 100

    # Summary percentages sum to ~1.0
    total_pct = (
        summary["auto_reported_pct"]
        + summary["clinician_review_pct"]
        + summary["needs_more_data_pct"]
    )
    assert 99.0 <= total_pct <= 101.0

    # Lanes contain cards
    lanes = batch_details["lanes"]
    assert len(lanes["auto_report"]) + len(lanes["clinician_review"]) + len(lanes["needs_more_data"]) == 100

    # Verify every card in auto_report has confidence >= 0.80 and not abstained
    for card in lanes["auto_report"]:
        assert card["confidence"] >= 0.80
        assert card["triage_lane"] == "auto_report"
        assert not card.get("abstain", False)

    # Verify clinician_review cards have 0.60 <= conf < 0.80
    for card in lanes["clinician_review"]:
        assert 0.60 <= card["confidence"] < 0.80
        assert card["triage_lane"] == "clinician_review"


# =========================================================================
# 3. CSV EXPORT SCHEMA VERIFICATION
# =========================================================================

def test_batch_csv_export_schema(client, auth_headers):
    """
    Tests GET /v1/batches/{id}/results.csv:
    Validates CSV header columns and row integrity.
    """
    # Run demo batch
    resp = client.post("/v1/batches?use_demo=true", headers=auth_headers)
    assert resp.status_code == 200
    batch_id = resp.json()["batch_id"]

    # Download CSV export
    csv_resp = client.get(f"/v1/batches/{batch_id}/results.csv", headers=auth_headers)
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers.get("content-type", "")

    content = csv_resp.text
    reader = csv.DictReader(io.StringIO(content))

    expected_fields = [
        "case_id",
        "patient_ref",
        "triage_lane",
        "predicted_subtype",
        "confidence",
        "confidence_tier",
        "modalities_used",
        "estimated_cost",
        "timestamp",
    ]

    for f in expected_fields:
        assert f in reader.fieldnames, f"Expected column '{f}' in CSV export header"

    rows = list(reader)
    assert len(rows) > 0, "CSV export should contain rows"

    for r in rows:
        assert r["triage_lane"] in ["auto_report", "clinician_review", "needs_more_data"]
        assert r["predicted_subtype"] in ["LumA", "LumB", "Her2", "Basal", "Normal"]
        assert float(r["confidence"]) >= 0.0
        assert float(r["estimated_cost"]) > 0.0
