"""API contract tests. Uses an empty serving dir (503) and optional real artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[2]
REAL_SERVING = REPO / "data" / "serving" / "patients.csv"


@pytest.fixture
def client_unloaded(tmp_path, monkeypatch):
    monkeypatch.setenv("BIONEXUS_SERVING_DIR", str(tmp_path))
    monkeypatch.setenv("BIONEXUS_ALLOW_MOCK", "false")
    monkeypatch.setenv("BIONEXUS_DEBUG", "true")
    from bio_nexus.serving import api as api_mod

    with TestClient(api_mod.app) as client:
        yield client


def test_health_unloaded(client_unloaded):
    r = client_unloaded.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "unloaded"
    assert body["model_loaded"] is False
    assert body["missing_artifacts"]
    assert body["allow_mock"] is False


def test_root_health_endpoint(client_unloaded):
    r = client_unloaded.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "model_loaded" in body
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"


def test_predict_503(client_unloaded):
    r = client_unloaded.post("/api/predict", json={"patient_id": "TCGA-XX-0000"})
    assert r.status_code == 503
    assert r.json()["error"] == "artifacts_not_loaded"


def test_explain_503(client_unloaded):
    r = client_unloaded.get("/api/explain/TCGA-XX-0000")
    assert r.status_code == 503


def test_graph_503(client_unloaded):
    r = client_unloaded.get("/api/graph")
    assert r.status_code == 503


def test_profile_503(client_unloaded):
    r = client_unloaded.get("/api/profile/TCGA-XX-0000")
    assert r.status_code == 503


def test_metrics_503(client_unloaded):
    r = client_unloaded.get("/api/metrics")
    assert r.status_code == 503


def test_cohort_503(client_unloaded):
    r = client_unloaded.get("/api/cohort")
    assert r.status_code == 503


def test_predict_422_missing_body(client_unloaded):
    r = client_unloaded.post("/api/predict", json={})
    assert r.status_code == 422


@pytest.mark.skipif(not REAL_SERVING.exists(), reason="data/serving artifacts not exported")
def test_serving_honest_unloaded_model():
    os.environ["BIONEXUS_SERVING_DIR"] = str(REPO / "data" / "serving")
    os.environ["BIONEXUS_ALLOW_LOCAL_FALLBACK"] = "false"
    os.environ["BIONEXUS_MODEL_URI"] = "models:/mosaic-subtype-classifier/Production"
    from importlib import reload

    from bio_nexus.serving import api as api_mod

    reload(api_mod)
    with TestClient(api_mod.app) as client:
        # /health reports model_loaded: false
        h = client.get("/health")
        assert h.status_code == 200
        h_data = h.json()
        assert h_data["model_loaded"] is False
        assert h_data["status"] == "degraded"

        # /model-info reports model_loaded: false and displays the load error
        info = client.get("/model-info")
        assert info.status_code == 200
        info_data = info.json()
        assert info_data["model_loaded"] is False
        assert info_data["model_load_error"] is not None

        # /predict returns 503 because no model is registered/loaded
        p = client.post("/predict", json={"patient_id": "TCGA-3C-AAAU"})
        assert p.status_code == 503
        err = p.json()
        error_name = err.get("error") or err.get("detail", {}).get("error")
        assert error_name == "model_not_loaded"


@pytest.mark.skipif(not REAL_SERVING.exists(), reason="data/serving artifacts not exported")
def test_live_oof_and_graph_646():
    os.environ["BIONEXUS_SERVING_DIR"] = str(REPO / "data" / "serving")
    os.environ["BIONEXUS_DEBUG"] = "true"
    os.environ["BIONEXUS_ALLOW_LOCAL_FALLBACK"] = "true"
    from importlib import reload

    from bio_nexus.serving import api as api_mod

    reload(api_mod)
    with TestClient(api_mod.app) as client:
        health = client.get("/api/health").json()
        assert health["mode"] == "live"
        assert health["model_loaded"] is True
        patients = client.get("/api/patients").json()
        assert len(patients) == 646
        pid = patients[0]["id"]
        p1 = client.post("/api/predict", json={"patient_id": pid}).json()
        p2 = client.post("/api/predict", json={"patient_id": pid}).json()
        assert p1 == p2
        assert p1["prediction_type"] == "out_of_fold"
        assert abs(sum(p1["probabilities"].values()) - 1) < 1e-3
        assert abs(sum(p1["modality_weights"].values()) - 1) < 1e-3
        unknown = client.post("/api/predict", json={"patient_id": "NOT-A-PATIENT"})
        assert unknown.status_code == 404
        g = client.get("/api/graph", params={"layer": "rna"}).json()
        assert g["stats"]["nodes"] == 646
        expl_a = client.get(f"/api/explain/{pid}", params={"layer": "rna"})
        if expl_a.status_code == 200:
            pid2 = patients[-1]["id"]
            expl_b = client.get(f"/api/explain/{pid2}", params={"layer": "rna"})
            if expl_b.status_code == 200:
                genes_a = [b["gene"] for b in expl_a.json()["biomarkers"]]
                genes_b = [b["gene"] for b in expl_b.json()["biomarkers"]]
                assert genes_a != genes_b or (
                    expl_a.json()["biomarkers"][0]["attribution"]
                    != expl_b.json()["biomarkers"][0]["attribution"]
                )
        metrics = client.get("/api/metrics").json()
        oof_acc = metrics["accuracy"]
        import pandas as pd
        from sklearn.metrics import accuracy_score

        oof = pd.read_parquet(REPO / "data" / "serving" / "oof_predictions.parquet")
        pats = pd.read_csv(REPO / "data" / "serving" / "patients.csv")
        merged = oof.merge(pats, on="patient_id")
        # Provisional baseline validation accuracy from plumbing files (0.4334)
        assert recomputed > 0.40
        prof = client.get(f"/api/profile/{pid}", params={"layer": "rna", "top_n": 5}).json()
        assert isinstance(prof["features"][0]["value"], float)
        assert "values" not in prof

        causal = client.get("/api/causal").json()
        assert causal["precomputed"] is True
        assert causal["patients_top_k_beats_random"] == 41
        assert causal["wilcoxon_p_value"] < 0.001

        fd = client.get("/api/fusion-diagnosis").json()
        assert fd["mean_rna_weight_incorrect"] > fd["mean_rna_weight_correct"]

        cohort = client.get("/api/cohort").json()
        assert cohort["total_patients"] == 646
        assert len(cohort["funnel"]) == 3
        assert cohort["funnel"][0]["count"] == 1035
        assert cohort["funnel"][1]["count"] == 733
        assert cohort["funnel"][2]["count"] == 646

        # --- Graph Explorer Tests ---
        g_summary = client.get("/api/graphs/summary").json()
        assert g_summary["rna"]["nodes"] == 646
        assert g_summary["rna"]["edges"] >= 10000
        assert g_summary["rna"]["min_degree"] >= 10
        assert g_summary["rna"]["neighbor_label_purity"]["overall"] > 0.4
        assert g_summary["mirna"]["nodes"] == 646
        assert g_summary["meth"]["nodes"] == 646

        # Ego graph (hop 1)
        ego1 = client.get(f"/api/graphs/rna/patient/{pid}?hops=1").json()
        assert ego1["patient_id"] == pid
        assert ego1["center_node"]["patient_id"] == pid
        assert len(ego1["nodes"]) >= 10
        assert len(ego1["edges"]) >= 10
        assert ego1["stats"]["same_subtype_fraction"] >= 0.0
        assert ego1["stats"]["truncated"] is False

        # Ego graph (hop 2)
        ego2 = client.get(f"/api/graphs/rna/patient/{pid}?hops=2").json()
        assert len(ego2["nodes"]) >= len(ego1["nodes"])
        assert len(ego2["nodes"]) <= 100

        # Multi-layer Compare
        comp = client.get(f"/api/graphs/patient/{pid}/compare").json()
        assert comp["patient_id"] == pid
        assert "rna" in comp["layers"]
        assert "mirna" in comp["layers"]
        assert "meth" in comp["layers"]
        assert "all_three" in comp["overlap_counts"]
        assert isinstance(comp["consensus_neighbors"], list)

        # Overview
        overview = client.get("/api/graphs/rna/overview?max_nodes=150").json()
        assert len(overview["nodes"]) == 150
        assert overview["thinned_for_display"] is True

