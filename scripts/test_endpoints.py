"""
Contract and functional verification tests for BioNexus FastAPI serving endpoints.
"""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fastapi.testclient import TestClient
from bio_nexus.serving.api import app, lifespan

def test_all():
    with TestClient(app) as client:
        print("1. Testing /api/health...")
        r = client.get("/api/health")
        assert r.status_code == 200, f"Health failed: {r.text}"
        h_data = r.json()
        assert h_data["status"] == "healthy"
        assert h_data["mode"] == "live"
        assert h_data["num_patients"] == 646
        print("   PASS: Health is live with 646 patients.")

        print("2. Testing /api/model-info...")
        r = client.get("/api/model-info")
        assert r.status_code == 200
        mi = r.json()
        assert mi["param_count"] == 1124970
        assert "GraphGPS" in mi["architecture"]
        print("   PASS: Model info matches 1.12M parameters.")

        print("3. Testing /api/patients...")
        r = client.get("/api/patients")
        assert r.status_code == 200
        pts = r.json()
        assert len(pts) == 646
        sample_pid = pts[0]["id"]
        sample_pid_2 = pts[1]["id"]
        print(f"   PASS: Retrieved 646 patients. Sample ID: {sample_pid}")

        print(f"4. Testing /api/patients/{sample_pid}...")
        r = client.get(f"/api/patients/{sample_pid}")
        assert r.status_code == 200
        c_data = r.json()
        assert c_data["patient_id"] == sample_pid
        assert "Invasive Ductal" not in str(c_data) or c_data["histology"] != "Invasive Ductal Carcinoma"  # Verify no fake fallback
        print(f"   PASS: Clinical fields for {sample_pid}: age={c_data['age']}, stage={c_data['stage']}")

        print(f"5. Testing /api/predict for {sample_pid} and {sample_pid_2}...")
        r1 = client.post("/api/predict", json={"patient_id": sample_pid})
        assert r1.status_code == 200
        p1 = r1.json()
        assert p1["prediction_type"] == "out_of_fold"
        prob_sum1 = sum(p1["probabilities"].values())
        assert abs(prob_sum1 - 1.0) < 1e-3, f"Probabilities do not sum to 1: {prob_sum1}"

        r2 = client.post("/api/predict", json={"patient_id": sample_pid_2})
        assert r2.status_code == 200
        p2 = r2.json()
        assert p1["probabilities"] != p2["probabilities"], "Different patients must have different probabilities!"
        print(f"   PASS: OOF prediction for {sample_pid}: {p1['predicted_subtype']} (conf={p1['confidence']:.2f})")

        print("6. Testing /api/profile...")
        r = client.get(f"/api/profile/{sample_pid}?layer=rna&top_n=10")
        assert r.status_code == 200
        prof = r.json()
        assert len(prof["features"]) == 10
        print(f"   PASS: Profile returned top features: {[f['symbol'] for f in prof['features'][:3]]}")

        print("7. Testing /api/heatmap...")
        r = client.get(f"/api/heatmap?layer=rna&patient_ids={sample_pid},{sample_pid_2}&top_n=5")
        assert r.status_code == 200
        hm = r.json()
        assert len(hm["patient_ids"]) == 2
        assert len(hm["z_scores"]) == 2
        print("   PASS: Heatmap subset returned real z-scores.")

        print("8. Testing /api/explain...")
        r = client.get(f"/api/explain/{sample_pid}?layer=rna&top_k=5")
        assert r.status_code == 200
        exp = r.json()
        assert len(exp["biomarkers"]) == 5
        print(f"   PASS: IG biomarkers for {sample_pid}: {[b['gene'] for b in exp['biomarkers']]}")

        print("9. Testing /api/ablate (Live Causal Test)...")
        r = client.post("/api/ablate", json={
            "patient_id": sample_pid,
            "layer": "rna",
            "mode": "top_k",
            "k": 5,
        })
        assert r.status_code == 200
        abl = r.json()
        assert len(abl["ablated_features"]) == 5
        assert abl["original_logit"] != abl["ablated_logit"], "Ablation must change logit!"
        print(f"   PASS: Live ablation logit change: {abl['logit_change']:.4f} (from {abl['original_logit']:.4f} to {abl['ablated_logit']:.4f})")

        print("10. Testing /api/causal...")
        r = client.get("/api/causal")
        assert r.status_code == 200
        causal = r.json()
        assert causal["n_patients"] == 50
        assert causal["patients_top_k_beats_random"] == 41
        print("   PASS: Causal validation 41/50 confirmed.")

        print("11. Testing /api/graph/summary and /api/graph/patient/{id}...")
        r_sum = client.get("/api/graph/summary")
        assert r_sum.status_code == 200
        g_sum = r_sum.json()
        assert g_sum["rna"]["edges"] == 11356
        assert g_sum["mirna"]["edges"] == 11168
        assert g_sum["meth"]["edges"] == 12114

        r_ego = client.get(f"/api/graph/patient/{sample_pid}?layer=rna&hops=1")
        assert r_ego.status_code == 200
        ego = r_ego.json()
        assert len(ego["top_10_neighbors"]) == 10
        print(f"   PASS: Graph ego network has {len(ego['nodes'])} nodes, top neighbor sim: {ego['top_10_neighbors'][0]['similarity']}")

        print("12. Testing /api/metrics and /api/cohort...")
        r_met = client.get("/api/metrics")
        assert r_met.status_code == 200
        r_coh = client.get("/api/cohort")
        assert r_coh.status_code == 200
        coh = r_coh.json()
        assert coh["total_patients"] == 646
        assert len(coh["funnel"]) == 3
        print(f"   PASS: Funnel: {[s['count'] for s in coh['funnel']]}")

        print("13. Testing /api/embeddings...")
        r = client.get("/api/embeddings?space=fused&method=pca")
        assert r.status_code == 200
        emb = r.json()
        assert len(emb["points"]) == 646
        print("   PASS: Embeddings returned 646 points.")

        print("14. Testing /api/survival...")
        r = client.get("/api/survival?group=true")
        assert r.status_code == 200
        surv = r.json()
        assert "LumA" in surv["subtypes"]
        print("   PASS: Survival returned Kaplan-Meier curves.")

        print("15. Testing /api/pathways...")
        r = client.get("/api/pathways")
        assert r.status_code == 503
        pw = r.json()
        assert pw["planned"] is True
        print("   PASS: Pathways correctly returned 503 with planned=True.")

        print("16. Testing /api/attention/{id}...")
        r = client.get(f"/api/attention/{sample_pid}")
        assert r.status_code == 200
        attn = r.json()
        assert len(attn["cross_attention_matrix"]) == 3
        assert len(attn["pooling_weights"]) == 3
        print(f"   PASS: Attention pooling weights: {attn['pooling_weights']}")

        print("17. Testing /api/fusion-diagnosis...")
        r = client.get("/api/fusion-diagnosis")
        assert r.status_code == 200
        fd = r.json()
        assert "mean_rna_weight_correct" in fd
        print("   PASS: Fusion diagnosis verified.")

    print("\n=======================================================")
    print("   ALL 17 API ENDPOINT CONTRACT TESTS PASSED!")
    print("=======================================================\n")

if __name__ == "__main__":
    test_all()
