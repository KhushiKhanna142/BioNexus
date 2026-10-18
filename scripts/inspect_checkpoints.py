import torch
from pathlib import Path

ckpt_dir = Path("data/processed/checkpoints")
files = ["best_model.pt", "fold_1.pt", "fold_2.pt", "fold_3.pt", "fold_4.pt", "fold_5.pt"]

print("=== CHECKPOINT INSPECTION REPORT ===")
for f in files:
    path = ckpt_dir / f
    if not path.exists():
        print(f"[-] {f}: NOT FOUND")
        continue
    o = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n[*] File: {f} | Size: {path.stat().st_size / (1024*1024):.2f} MB | Type: {type(o).__name__}")
    if isinstance(o, dict):
        keys = list(o.keys())
        print(f"    Total keys: {len(keys)}")
        is_state_dict = all(isinstance(v, torch.Tensor) for v in o.values())
        print(f"    Is pure state_dict (weights only): {is_state_dict}")
        print("    Sample keys & shapes:")
        for k in keys[:10]:
            v = o[k]
            sh = getattr(v, "shape", None)
            dtype = getattr(v, "dtype", type(v).__name__)
            print(f"      {k} -> shape={sh}, dtype={dtype}")
    else:
        print(f"    Loaded object: {repr(o)[:200]}")
