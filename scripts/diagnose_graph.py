import sys
sys.path.insert(0, "src")
from bio_nexus.serving.api import app
from fastapi.testclient import TestClient
import json

with TestClient(app) as client:
    health = client.get("/api/health").json()
    print("HEALTH:", json.dumps(health, indent=2))
    
    for layer in ["rna", "mirna", "meth"]:
        res = client.get(f"/api/graphs/{layer}/patient/TCGA-3C-AALI?hops=1")
        print(f"\nEGO {layer} status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            print(f"  Nodes count: {len(data['nodes'])}, Edges count: {len(data['edges'])}")
            print("  Center node:", data.get("center_node"))
            print("  Top 10 neighbors count:", len(data.get("top_10_neighbors") or []))
            print("  Sample top 10 neighbors:", data.get("top_10_neighbors")[:3] if data.get("top_10_neighbors") else [])
            print("  Graph stats details:", data.get("graph_stats_details"))
            print("  Clinical details:", data.get("clinical_details"))

    comp = client.get("/api/graphs/patient/TCGA-3C-AALI/compare")
    print(f"\nCOMPARE status: {comp.status_code}")
    if comp.status_code == 200:
        cdata = comp.json()
        print("  Compare keys:", list(cdata.keys()))
        for lk, lv in cdata["layers"].items():
            print(f"    {lk}: count={lv['count']}, same_subtype_fraction={lv['same_subtype_fraction']}, mean_sim={lv['mean_similarity']}")
