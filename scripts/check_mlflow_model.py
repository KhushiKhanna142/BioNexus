import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sqlite3
import mlflow
import mlflow.pytorch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
db_path = REPO_ROOT / "mlflow.db"

print("=== CHECKING MLFLOW DB MODEL VERSIONS ===")
con = sqlite3.connect(db_path)
cur = con.cursor()
rows = cur.execute("SELECT name, version, current_stage, source FROM model_versions").fetchall()
for r in rows:
    print(" ", r)

print("\n=== TESTING MLFLOW MODEL LOADING ===")
mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")
model_uri = "models:/mosaic-subtype-classifier/Production"
try:
    m = mlflow.pytorch.load_model(model_uri)
    print(f"[PASS] Successfully loaded model from {model_uri}!")
    print(f"  Model type: {type(m).__name__}")
except Exception as e:
    print(f"[FAIL] Error loading model from {model_uri}: {e}")
