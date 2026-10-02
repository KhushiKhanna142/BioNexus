"""
bio_nexus.security.cohort_manager
---------------------------------
Manages cohort-level cryptographic Merkle batch anchoring for all 646 patients.
Enables a single Polygon on-chain transaction to anchor the entire multi-omics
dataset, while allowing zero-gas, sub-millisecond O(log N) verification of any
individual patient record.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from bio_nexus.security.merkle_tree import MerkleTree, compute_patient_leaf
from bio_nexus.security.patient_vault import PatientVault
from bio_nexus.security.polygon_service import PolygonService


class CohortBatchManager:
    def __init__(self, storage_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[3]
        self.storage_dir = storage_dir or (base_dir / "data" / "security")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.batch_file = self.storage_dir / "cohort_merkle_batch.json"

        self.vault = PatientVault()
        self.polygon = PolygonService()
        self._tree: MerkleTree | None = None
        self._batch_metadata: dict[str, Any] | None = None
        self._patient_leaves: dict[str, dict[str, Any]] = {}

        self._load_saved_batch()

    def _load_saved_batch(self) -> None:
        if self.batch_file.exists():
            try:
                data = json.loads(self.batch_file.read_text())
                self._batch_metadata = data.get("metadata", {})
                self._patient_leaves = data.get("leaves", {})
                leaves_list = [v["leaf_hash"] for v in self._patient_leaves.values()]
                keys_list = list(self._patient_leaves.keys())
                if leaves_list:
                    self._tree = MerkleTree(leaves_list, keys_list)
            except Exception as e:
                print(f"[CohortBatchManager] Warning loading batch: {e}")

    def build_and_anchor_cohort(self, store: Any, get_canonical_func: Any = None) -> dict[str, Any]:
        """
        Builds Merkle Tree for all 646 cohort patients and anchors root to Polygon.
        """
        leaves: list[str] = []
        keys: list[str] = []
        patient_map: dict[str, dict[str, Any]] = {}

        patients_df = store.patients if hasattr(store, "patients") else store

        # 1. Process all patients deterministically
        for _, row in patients_df.iterrows():
            pid = str(row["patient_id"])
            p_hash = self.vault.anonymize_id(pid)

            if get_canonical_func is not None:
                record = get_canonical_func(store, pid)
                fp = self.vault.compute_fingerprint(record)
                subtype = str(record.get("predicted_subtype", "LumA"))
                conf = float(record.get("confidence", 0.9500))
            else:
                subtype = str(row.get("true_pam50", "LumA"))
                fold_id = int(row.get("fold_id", 0))
                canonical_summary = {
                    "patient_id": pid,
                    "true_pam50": subtype,
                    "fold_id": fold_id,
                }
                fp = self.vault.compute_fingerprint(canonical_summary)
                conf = 0.9500

            leaf = compute_patient_leaf(p_hash, fp, subtype, conf)
            leaves.append(leaf)
            keys.append(pid)

            patient_map[pid] = {
                "patient_id": pid,
                "patient_hash": p_hash,
                "data_fingerprint": fp,
                "predicted_subtype": subtype,
                "confidence": conf,
                "leaf_hash": leaf,
            }


        # 2. Build Merkle Tree
        tree = MerkleTree(leaves, keys)
        self._tree = tree
        self._patient_leaves = patient_map

        # 3. Anchor Merkle Root to Polygon Blockchain
        anchor_receipt = self.polygon.anchor_cohort_root(
            merkle_root=tree.root,
            total_records=len(keys),
            cohort_name="TCGA-BRCA",
        )

        metadata = {
            "cohort_name": "TCGA-BRCA",
            "total_patients": len(keys),
            "merkle_root": tree.root,
            "tree_depth": len(tree.levels) - 1,
            "timestamp": int(time.time()),
            "on_chain_receipt": anchor_receipt,
        }
        self._batch_metadata = metadata

        # 4. Save to persistent file
        self.batch_file.write_text(json.dumps({
            "metadata": metadata,
            "leaves": patient_map,
        }, indent=2))

        return {
            "status": "success",
            "merkle_root": tree.root,
            "total_patients": len(keys),
            "tree_depth": len(tree.levels) - 1,
            "receipt": anchor_receipt,
        }

    def get_status(self) -> dict[str, Any]:
        """
        Returns the overall cohort batch status.
        """
        if not self._batch_metadata or not self._tree:
            return {
                "is_batch_anchored": False,
                "total_patients": 646,
                "message": "Cohort Merkle Tree has not been anchored yet.",
            }

        return {
            "is_batch_anchored": True,
            "merkle_root": self._batch_metadata.get("merkle_root"),
            "total_patients": self._batch_metadata.get("total_patients", 646),
            "tree_depth": self._batch_metadata.get("tree_depth", 10),
            "timestamp": self._batch_metadata.get("timestamp"),
            "on_chain_receipt": self._batch_metadata.get("on_chain_receipt", {}),
        }

    def get_patient_proof(self, patient_id: str) -> dict[str, Any]:
        """
        Generates Merkle proof for a specific patient in O(log N) time.
        """
        if not self._tree or patient_id not in self._patient_leaves:
            return {
                "has_proof": False,
                "error": "Patient not found in anchored cohort tree.",
            }

        leaf_info = self._patient_leaves[patient_id]
        proof = self._tree.get_proof_for_key(patient_id)
        root = self._tree.root

        is_valid = MerkleTree.verify_proof(leaf_info["leaf_hash"], proof, root)

        return {
            "has_proof": True,
            "patient_id": patient_id,
            "leaf_hash": leaf_info["leaf_hash"],
            "merkle_root": root,
            "proof_path": proof,
            "proof_length": len(proof),
            "is_valid": is_valid,
            "receipt": self._batch_metadata.get("on_chain_receipt", {}) if self._batch_metadata else {},
        }

    def verify_patient_leaf(self, patient_id: str, current_leaf: str) -> dict[str, Any]:
        """
        Verifies if an arbitrary or modified patient leaf matches the cohort Merkle Root.
        Used for instantaneous zero-gas tamper detection!
        """
        if not self._tree or patient_id not in self._patient_leaves:
            return {
                "is_verified": False,
                "error": "Cohort Merkle Tree not initialized for patient.",
            }

        proof = self._tree.get_proof_for_key(patient_id)
        root = self._tree.root
        is_valid = MerkleTree.verify_proof(current_leaf, proof, root)

        return {
            "patient_id": patient_id,
            "current_leaf": current_leaf,
            "merkle_root": root,
            "is_verified": is_valid,
            "status": "VERIFIED_AUTHENTIC" if is_valid else "TAMPER_DETECTED",
            "proof_length": len(proof),
        }


_COHORT_MANAGER_INSTANCE: CohortBatchManager | None = None


def get_cohort_manager() -> CohortBatchManager:
    global _COHORT_MANAGER_INSTANCE
    if _COHORT_MANAGER_INSTANCE is None:
        _COHORT_MANAGER_INSTANCE = CohortBatchManager()
    return _COHORT_MANAGER_INSTANCE
