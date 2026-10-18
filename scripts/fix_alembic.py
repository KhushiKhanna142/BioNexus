import sqlite3
import shutil
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
db_path = repo_root / "mlflow.db"
target_copy = repo_root / "mlruns" / "mlflow.db"

con = sqlite3.connect(db_path)
con.execute("UPDATE alembic_version SET version_num = '5b0e9adcef9c'")
con.commit()
rows = con.execute("SELECT * FROM alembic_version").fetchall()
con.close()

shutil.copyfile(db_path, target_copy)
print("SUCCESS: Updated alembic revision to:", rows)
