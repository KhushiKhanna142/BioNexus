"""
tests/test_cost_accuracy.py
---------------------------
Tests for Cost vs Accuracy simulation feature:
1. Curve monotonicity checks:
   - Increasing confidence threshold tau increases or preserves average cost (monotonic cost escalation).
   - Increasing average budget increases or preserves macro-F1 and accuracy.
   - Bootstrap confidence intervals are well-ordered: ci_lower <= mean <= ci_upper.
2. Slider mapping:
   - Budget slider maps to the nearest precomputed point.
   - Confidence-threshold slider maps to the nearest precomputed point.
   - Edge cases (below min budget, above max budget) resolve cleanly.
3. Empty state handling:
   - When results/f3/ artifacts are missing or unreadable, endpoint gracefully returns
     status="empty" without 500 error or crashing.
"""

import shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from bio_nexus.serving.api import app
from bio_nexus.serving.v1_router import RESULTS_DIR, find_nearest_policy_point


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "dev-admin-key"}


@pytest.fixture
def client():
    return TestClient(app)



def test_cost_accuracy_monotonicity(client, auth_headers):
    """Verify that precomputed policy curve satisfies strict monotonicity requirements."""
    resp = client.get("/v1/simulate/cost-accuracy", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["empty"] is False
    points = data["policy_points"]
    assert len(points) >= 5, "Expected at least 5 points along policy curve"

    # Check monotonicity along thresholds
    prev_threshold = -1.0
    prev_cost = -1.0
    prev_f1 = -1.0
    prev_acc = -1.0

    for pt in points:
        t = pt["threshold"]
        c = pt["avg_cost"]
        f1 = pt["macro_f1"]
        acc = pt["accuracy"]

        # Threshold increases monotonically
        assert t >= prev_threshold, f"Threshold {t} not >= {prev_threshold}"
        # Cost increases monotonically with threshold (higher confidence requires escalating more cases)
        assert c >= prev_cost, f"Cost {c} not >= {prev_cost}"
        # Macro-F1 increases monotonically with cost
        assert f1 >= prev_f1, f"Macro-F1 {f1} not >= {prev_f1}"
        # Accuracy increases monotonically with cost
        assert acc >= prev_acc, f"Accuracy {acc} not >= {prev_acc}"

        # Bootstrap confidence intervals check
        assert pt["macro_f1_ci_lower"] <= f1 <= pt["macro_f1_ci_upper"]
        assert pt["accuracy_ci_lower"] <= acc <= pt["accuracy_ci_upper"]

        prev_threshold = t
        prev_cost = c
        prev_f1 = f1
        prev_acc = acc


def test_slider_nearest_point_mapping(client, auth_headers):
    """Verify budget and threshold slider values map to nearest precomputed points."""
    # 1. Budget slider query
    resp = client.get("/v1/simulate/cost-accuracy?budget=800", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    nearest = data["nearest_point"]
    assert nearest is not None
    assert abs(nearest["avg_cost"] - 800) <= 50
    assert "At budget $" in data["readout"]
    assert "% of full-panel accuracy" in data["readout"]

    # 2. Threshold slider query
    resp_thresh = client.get("/v1/simulate/cost-accuracy?threshold=0.75", headers=auth_headers)
    assert resp_thresh.status_code == 200
    data_t = resp_thresh.json()
    nearest_t = data_t["nearest_point"]
    assert nearest_t is not None
    assert abs(nearest_t["threshold"] - 0.75) < 0.05

    # 3. Direct helper unit tests
    points = [
        {"threshold": 0.50, "avg_cost": 520.0},
        {"threshold": 0.75, "avg_cost": 745.0},
        {"threshold": 0.90, "avg_cost": 1010.0},
    ]
    # Under min budget maps to lowest point
    low = find_nearest_policy_point(points, target_budget=300.0)
    assert low["avg_cost"] == 520.0

    # Over max budget maps to highest point
    high = find_nearest_policy_point(points, target_budget=2000.0)
    assert high["avg_cost"] == 1010.0

    # Threshold mapping
    mid = find_nearest_policy_point(points, target_threshold=0.73)
    assert mid["threshold"] == 0.75


def test_empty_state_when_results_missing(client, auth_headers, tmp_path, monkeypatch):
    """Verify graceful empty state response when results/f3/ files do not exist."""
    # Temporarily point RESULTS_DIR to empty directory
    fake_empty_dir = tmp_path / "empty_results"
    fake_empty_dir.mkdir()

    import bio_nexus.serving.v1_router as router_mod
    monkeypatch.setattr(router_mod, "RESULTS_DIR", fake_empty_dir)

    resp = client.get("/v1/simulate/cost-accuracy", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["empty"] is True
    assert data["status"] == "empty"
    assert "missing" in data["message"].lower()
    assert data["curve"] == []
    assert data["policy_points"] == []
    assert data["modality_combinations"] == []


def test_headline_includes_placeholder_caveat(client, auth_headers):
    """Verify code-generated headline contains performance readout and placeholder caveat."""
    resp = client.get("/v1/simulate/cost-accuracy", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()

    headline = data["headline"]
    assert "BioNexus adaptive policy" in headline or "adaptive policy" in headline
    assert "accuracy" in headline.lower() or "cost" in headline.lower()
    assert "placeholder" in headline.lower(), "Headline must contain placeholder caveat"

    # Assay cost table badges
    gain_table = data["gain_table"]
    assert gain_table["is_placeholder"] is True
    for assay in gain_table.get("assays", []):
        assert assay["is_placeholder"] is True
        assert assay["badge"] == "Placeholder Quote"
