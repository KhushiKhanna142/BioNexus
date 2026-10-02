"""
bio_nexus.security.merkle_tree
------------------------------
Cryptographic Merkle Tree implementation for multi-omics cohort batch anchoring.
Allows all 646+ patients to be anchored onto the Polygon blockchain in a single
O(1) transaction, with O(log N) zero-gas individual verification proofs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_pair(left_hex: str, right_hex: str) -> str:
    """
    OpenZeppelin-style sorted canonical pairwise hash.
    Ensures proof verification does not need direction flags.
    """
    l_bytes = bytes.fromhex(left_hex.replace("0x", ""))
    r_bytes = bytes.fromhex(right_hex.replace("0x", ""))
    if l_bytes <= r_bytes:
        combined = l_bytes + r_bytes
    else:
        combined = r_bytes + l_bytes
    return hashlib.sha256(combined).hexdigest()


def compute_patient_leaf(
    patient_hash: str,
    data_fingerprint: str,
    predicted_subtype: str | int,
    confidence: float | int,
) -> str:
    """
    Computes a deterministic cryptographic leaf for a patient.
    Leaf = SHA-256(patient_hash + data_fingerprint + subtype + confidence)
    """
    clean_ph = patient_hash.replace("0x", "").lower()
    clean_df = data_fingerprint.replace("0x", "").lower()
    payload = f"{clean_ph}:{clean_df}:{predicted_subtype}:{confidence:.4f}"
    return sha256_hex(payload)


class MerkleTree:
    """
    Complete binary Merkle Tree supporting proof generation and verification.
    """

    def __init__(self, leaves: list[str], leaf_keys: list[str] | None = None):
        """
        leaves: list of 32-byte hex strings.
        leaf_keys: optional list of identifiers (e.g. patient_id or patient_hash)
                   corresponding 1:1 with leaves.
        """
        if not leaves:
            raise ValueError("Cannot build Merkle Tree with 0 leaves.")

        self.leaves = [l.replace("0x", "").lower() for l in leaves]
        self.leaf_keys = leaf_keys or []
        self.key_to_index: dict[str, int] = {k: i for i, k in enumerate(self.leaf_keys)}

        self.levels: list[list[str]] = []
        self._build_tree()

    def _build_tree(self) -> None:
        current_level = list(self.leaves)
        self.levels.append(current_level)

        while len(current_level) > 1:
            next_level: list[str] = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Odd number of leaves: duplicate the last node (Bitcoin standard)
                    right = left
                parent = hash_pair(left, right)
                next_level.append(parent)
            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> str:
        """Returns the 32-byte hex Merkle Root (prefixed with 0x)."""
        return "0x" + self.levels[-1][0]

    def get_proof(self, index: int) -> list[str]:
        """
        Generates the Merkle audit path (sibling hashes) for leaf at index.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range (0..{len(self.leaves)-1})")

        proof: list[str] = []
        current_idx = index

        for level in self.levels[:-1]:
            if current_idx % 2 == 0:
                # Left child, sibling is right
                sibling_idx = current_idx + 1 if current_idx + 1 < len(level) else current_idx
            else:
                # Right child, sibling is left
                sibling_idx = current_idx - 1

            sibling_hash = level[sibling_idx]
            proof.append("0x" + sibling_hash)
            current_idx //= 2

        return proof

    def get_proof_for_key(self, key: str) -> list[str]:
        if key not in self.key_to_index:
            raise KeyError(f"Key {key} not found in Merkle Tree.")
        return self.get_proof(self.key_to_index[key])

    @staticmethod
    def verify_proof(leaf_hex: str, proof: list[str], root_hex: str) -> bool:
        """
        Cryptographically verifies a Merkle proof in O(log N) time with 0 gas.
        """
        current = leaf_hex.replace("0x", "").lower()
        root = root_hex.replace("0x", "").lower()

        for sibling in proof:
            sib_clean = sibling.replace("0x", "").lower()
            current = hash_pair(current, sib_clean)

        return current == root

    def export_summary(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "total_leaves": len(self.leaves),
            "tree_depth": len(self.levels) - 1,
            "leaf_keys_count": len(self.leaf_keys),
        }
