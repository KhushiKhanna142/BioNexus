import unittest
import pandas as pd
import json

class TestInspect(unittest.TestCase):
    def test_inspect_data(self):
        pts = pd.read_csv('data/serving/patients.csv')
        print(f"\n[INSPECT] Patients count: {len(pts)}")
        print("[INSPECT] Subtype counts:")
        for k, v in pts['true_pam50'].value_counts().items():
            print(f"  {k}: {v}")
        
        rna = pd.read_parquet('data/serving/rna_matrix.parquet')
        print(f"[INSPECT] RNA shape: {rna.shape}")
        mirna = pd.read_parquet('data/serving/mirna_matrix.parquet')
        print(f"[INSPECT] miRNA shape: {mirna.shape}")
        meth = pd.read_parquet('data/serving/meth_matrix.parquet')
        print(f"[INSPECT] Meth shape: {meth.shape}")
