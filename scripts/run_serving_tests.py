"""
Direct serving contract verification script.
Tests all endpoints: /health, /api/health, /model-info, /predict
across both unloaded, honest error, and loaded states.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]

def run_tests():
    print("==========================================================")
    print("   RUNNING SERVING CONTRACT & HONEST HEALTH TESTS")
    print("==========================================================")
    
    # --- TEST SUITE 1: UNLOADED ARTIFACTS ---
    print("\n[TEST GROUP 1: UNLOADED ARTIFACTS]")
    os.environ["BIONEXUS_SERVING_DIR"] = str(REPO / "tests" / "fixtures")
    os.environ["BIONEXUS_ALLOW_MOCK"] = "false"
    os.environ["BIONEXUS_ALLOW_LOCAL_FALLBACK"] = "false"
    
    from bio_nexus.serving import api as api_mod
    with TestClient(api_mod.app) as client:
        # 1. /api/health unloaded
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "degraded"
        assert body["mode"] == "unloaded"
        assert body["model_loaded"] is False
        print("  [PASS] /api/health unloaded: status='degraded', model_loaded=False")

        # 2. /health root alias
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"
        print("  [PASS] /health root alias: status='degraded', model_loaded=False")

        # 3. /api/predict 503
        r = client.post("/api/predict", json={"patient_id": "TCGA-XX-0000"})
        assert r.status_code == 503
        assert r.json().get("error") == "artifacts_not_loaded" or r.json().get("detail", {}).get("error") == "artifacts_not_loaded"
        print("  [PASS] /api/predict returns 503 artifacts_not_loaded")

    # --- TEST SUITE 2: REAL ARTIFACTS LOADED, NO REGISTERED MODEL (HONEST SERVING) ---
    print("\n[TEST GROUP 2: REAL ARTIFACTS LOADED, NO REGISTERED MLFLOW MODEL]")
    os.environ["BIONEXUS_SERVING_DIR"] = str(REPO / "data" / "serving")
    os.environ["BIONEXUS_ALLOW_LOCAL_FALLBACK"] = "false"
    os.environ["BIONEXUS_MODEL_URI"] = "models:/mosaic-subtype-classifier/Production"
    
    from importlib import reload
    reload(api_mod)
    with TestClient(api_mod.app) as client:
        # 4. /health reports honest degraded status and model_loaded: false
        h = client.get("/health")
        assert h.status_code == 200
        h_data = h.json()
        assert h_data["status"] == "degraded"
        assert h_data["model_loaded"] is False
        assert h_data["model_load_error"] is not None
        print(f"  [PASS] /health honest report: status='{h_data['status']}', model_loaded={h_data['model_loaded']}")
        print(f"         model_load_error: {h_data['model_load_error']}")

        # 5. /model-info reports model_loaded: false and load error details
        info = client.get("/model-info")
        assert info.status_code == 200
        info_data = info.json()
        assert info_data["model_loaded"] is False
        assert info_data["model_load_error"] is not None
        assert info_data["model_uri"] == "models:/mosaic-subtype-classifier/Production"
        print(f"  [PASS] /model-info honest report: model_loaded=False, uri='{info_data['model_uri']}'")

        # 6. /predict returns 503 Service Unavailable with clear model_not_loaded error
        p = client.post("/predict", json={"patient_id": "TCGA-3C-AAAU"})
        assert p.status_code == 503, f"Expected 503, got {p.status_code}"
        p_data = p.json()
        err_code = p_data.get("error") or p_data.get("detail", {}).get("error")
        assert err_code == "model_not_loaded"
        print(f"  [PASS] POST /predict returns HTTP 503: error='{err_code}'")
        print(f"         message: '{p_data.get('detail', {}).get('message')}'")

        # 7. GET /predict/{id} returns 503 as well
        p_get = client.get("/predict/TCGA-3C-AAAU")
        assert p_get.status_code == 503
        assert (p_get.json().get("error") or p_get.json().get("detail", {}).get("error")) == "model_not_loaded"
        print("  [PASS] GET /predict/{patient_id} returns HTTP 503: model_not_loaded")

    # --- TEST SUITE 3: PROVISIONAL LOCAL FALLBACK MODEL LOADED ---
    print("\n[TEST GROUP 3: PROVISIONAL LOCAL FALLBACK MODEL]")
    os.environ["BIONEXUS_ALLOW_LOCAL_FALLBACK"] = "true"
    reload(api_mod)
    with TestClient(api_mod.app) as client:
        # 8. /health reports model_loaded: true
        h_fb = client.get("/health")
        assert h_fb.status_code == 200
        h_fb_data = h_fb.json()
        assert h_fb_data["status"] == "healthy"
        assert h_fb_data["model_loaded"] is True
        assert h_fb_data["model_source"] == "local_checkpoint_provisional"
        print(f"  [PASS] /health with fallback: status='{h_fb_data['status']}', source='{h_fb_data['model_source']}'")

        # 9. /predict returns 200 with prediction
        p_valid = client.post("/predict", json={"patient_id": "TCGA-3C-AAAU"})
        assert p_valid.status_code == 200
        pred_res = p_valid.json()
        assert pred_res["patient_id"] == "TCGA-3C-AAAU"
        assert "predicted_subtype" in pred_res
        assert "probabilities" in pred_res
        print(f"  [PASS] POST /predict returns HTTP 200: subtype='{pred_res['predicted_subtype']}', conf={pred_res['confidence']}")

    print("\n==========================================================")
    print("   ALL 9 SERVING CONTRACT TESTS PASSED PERFECTLY!")
    print("==========================================================")

if __name__ == "__main__":
    run_tests()
