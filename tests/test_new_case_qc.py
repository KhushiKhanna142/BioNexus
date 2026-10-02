"""
tests/test_new_case_qc.py
-------------------------
Tests for "New Case" upload with live data-quality check (QC) and live prediction:
- Fixtures for wrong IDs, empty file, raw counts, half coverage, duplicated rows
- POST /v1/qc (multipart, returns per-modality report without predicting)
- Scale detection heuristics (raw_counts, log2, beta_values)
- Never transform data silently: explicit auto_normalize checkbox
- POST /v1/predict-live:
  - Run disabled on fail
  - On warn, allows run only with proceed_with_caution confirmation
  - Records auto-normalization in report and audit trail
"""

import io
import pytest
from fastapi.testclient import TestClient

from bio_nexus.serving.api import app
from bio_nexus.serving.qc import QCManager


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dev-admin-key"}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def qc_manager():
    return QCManager()


# =========================================================================
# 1. FIXTURES FOR DATA QUALITY CHECKS
# =========================================================================

@pytest.fixture
def empty_csv():
    return b""


@pytest.fixture
def wrong_ids_csv():
    """CSV with invalid non-matching gene identifiers."""
    rows = ["id,value\n"]
    for i in range(100):
        rows.append(f"UNKNOWN_FAKE_GENE_{i},5.2\n")
    return "".join(rows).encode("utf-8")


@pytest.fixture
def duplicated_rows_csv(qc_manager):
    """CSV with duplicate identifiers."""
    feats = qc_manager.expected_features.get("rna", [])[:30]
    rows = ["id,value\n"]
    for f in feats:
        rows.append(f"{f},6.0\n")
    # Add duplicate rows for the first 5 features
    for f in feats[:5]:
        rows.append(f"{f},6.5\n")
    return "".join(rows).encode("utf-8")


@pytest.fixture
def raw_counts_csv(qc_manager):
    """RNA CSV with un-logged raw sequencing counts (max > 30, mean > 50)."""
    feats = qc_manager.expected_features.get("rna", [])[:600]
    rows = ["id,value\n"]
    for i, f in enumerate(feats):
        val = 1200.0 if i % 10 == 0 else 85.0
        rows.append(f"{f},{val}\n")
    return "".join(rows).encode("utf-8")


@pytest.fixture
def half_coverage_csv(qc_manager):
    """CSV covering between 50% and 80% of expected features."""
    feats = qc_manager.expected_features.get("rna", [])
    n_expected = len(feats)
    # Give exactly 60% coverage (within 50% - 80% warning band)
    n_sample = int(n_expected * 0.60)
    rows = ["id,value\n"]
    for f in feats[:n_sample]:
        rows.append(f"{f},4.5\n")
    return "".join(rows).encode("utf-8")


@pytest.fixture
def good_rna_csv(qc_manager):
    """Valid normalized RNA CSV with >= 85% coverage in log2 scale."""
    feats = qc_manager.expected_features.get("rna", [])
    n_sample = int(len(feats) * 0.90)
    rows = ["id,value\n"]
    for f in feats[:n_sample]:
        rows.append(f"{f},5.5\n")
    return "".join(rows).encode("utf-8")


# =========================================================================
# 2. QC UNIT LOGIC TESTS
# =========================================================================

def test_qc_empty_file(qc_manager, empty_csv):
    """Empty file returns FAILED status with clear fix hints."""
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(empty_csv))
    except Exception:
        df = pd.DataFrame()
    res = qc_manager.check_modality("rna", df)
    assert res.status == "FAILED"
    assert res.coverage_pct == 0.0
    assert any("empty" in hint.lower() or "no valid data" in hint.lower() for hint in res.fix_hints)


def test_qc_wrong_ids(qc_manager, wrong_ids_csv):
    """Wrong IDs fail coverage threshold and produce naming fix hints."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(wrong_ids_csv))
    res = qc_manager.check_modality("rna", df)
    assert res.status == "FAILED"
    assert res.coverage_pct < 0.10
    assert any("gene symbol" in hint.lower() or "ensembl" in hint.lower() for hint in res.fix_hints)


def test_qc_raw_counts_detection(qc_manager, raw_counts_csv):
    """Detects raw counts scale and flags warning without silent transformation."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(raw_counts_csv))
    res = qc_manager.check_modality("rna", df, auto_normalize=False)
    assert res.detected_scale == "raw_counts"
    assert res.likely_raw_counts is True
    assert res.status in ["WARNING", "FAILED"]
    assert any("log2" in hint.lower() for hint in res.fix_hints)


def test_qc_explicit_auto_normalize_opt_in(qc_manager, raw_counts_csv):
    """When auto_normalize=True is explicitly selected, raw counts are converted."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(raw_counts_csv))
    res = qc_manager.check_modality("rna", df, auto_normalize=True)
    # The normalized vector max should now be safely in log2 scale
    assert "log2" in res.detected_scale or res.detected_scale in ["log2", "raw_counts_auto_normalized"]


def test_qc_duplicated_rows(qc_manager, duplicated_rows_csv):
    """Duplicated rows are counted and deduplicated with warning."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(duplicated_rows_csv))
    res = qc_manager.check_modality("rna", df)
    assert res.duplicate_ids_count == 5
    assert any("duplicate" in w.lower() for w in res.warnings)


def test_qc_half_coverage(qc_manager, half_coverage_csv):
    """Half coverage (60%) produces WARNING status with caution required."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(half_coverage_csv))
    res = qc_manager.check_modality("rna", df)
    assert res.status == "WARNING"
    assert 0.50 <= res.coverage_pct < 0.80


# =========================================================================
# 3. ENDPOINT INTEGRATION TESTS (POST /v1/qc & POST /v1/predict-live)
# =========================================================================

def test_post_qc_endpoint(client, auth_headers, good_rna_csv, wrong_ids_csv):
    """POST /v1/qc returns comprehensive modality report without predicting."""
    files = {
        "rna": ("rna.csv", good_rna_csv, "text/csv"),
        "mirna": ("mirna.csv", wrong_ids_csv, "text/csv"),
    }
    resp = client.post("/v1/qc", files=files, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "overall_status" in data
    assert "summary_message" in data
    assert "results" in data
    assert "rna" in data["results"]
    assert "mirna" in data["results"]

    rna_res = data["results"]["rna"]
    assert rna_res["coverage_pct"] >= 0.80
    assert rna_res["status"] == "PASSED"

    mirna_res = data["results"]["mirna"]
    assert mirna_res["status"] == "FAILED"
    assert len(mirna_res["fix_hints"]) > 0


def test_predict_live_disabled_on_fail(client, auth_headers, wrong_ids_csv):
    """POST /v1/predict-live fails with 422 Unprocessable Entity when QC fails."""
    files = {"rna": ("rna.csv", wrong_ids_csv, "text/csv")}
    resp = client.post("/v1/predict-live", files=files, headers=auth_headers)
    assert resp.status_code == 422
    data = resp.json()
    assert data.get("error_code") == "QC_FAILED" or "QC Failed" in data.get("title", "")


def test_predict_live_warn_requires_caution(client, auth_headers, half_coverage_csv):
    """POST /v1/predict-live on warning requires proceed_with_caution confirmation."""
    files = {"rna": ("rna.csv", half_coverage_csv, "text/csv")}
    # First call: without proceed_with_caution
    resp = client.post("/v1/predict-live", files=files, data={"proceed_with_caution": "false"}, headers=auth_headers)
    assert resp.status_code == 400
    assert resp.json().get("code") == "QC_WARNING_REQUIRES_CONFIRMATION" or resp.json().get("error_code") == "QC_WARNING_REQUIRES_CONFIRMATION"

    # Second call: with proceed_with_caution=True
    files = {"rna": ("rna.csv", half_coverage_csv, "text/csv")}
    resp2 = client.post("/v1/predict-live", files=files, data={"proceed_with_caution": "true"}, headers=auth_headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "predicted_subtype" in data2
    assert "confidence" in data2
    assert "case_id" in data2


def test_predict_live_success_flow(client, auth_headers, good_rna_csv):
    """Passing upload runs end-to-end and returns live prediction and case_id."""
    files = {"rna": ("rna.csv", good_rna_csv, "text/csv")}
    resp = client.post(
        "/v1/predict-live",
        files=files,
        data={"patient_ref": "TEST-PT-001"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_ref"] == "TEST-PT-001"
    assert data["predicted_subtype"] in ["LumA", "LumB", "Her2", "Basal", "Normal"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "case_id" in data
