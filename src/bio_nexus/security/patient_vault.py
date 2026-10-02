"""
bio_nexus.security.patient_vault
---------------------------------
Client-side cryptographic vault for patient multi-omics records.
Provides:
1. One-way pseudonymous patient hashing (HIPAA/GDPR compliance).
2. AES-256-GCM authenticated encryption for off-chain storage.
3. Canonical cryptographic fingerprinting (SHA-256) for Polygon blockchain anchoring.
4. Tamper detection and local vault storage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class PatientVault:
    def __init__(self, vault_dir: Path | None = None, master_key_hex: str | None = None, salt: str | None = None):
        self.vault_dir = vault_dir or (Path(__file__).resolve().parents[3] / "data" / "security")
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = self.vault_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize or load Master 256-bit key
        key_file = self.vault_dir / "vault_master.key"
        if master_key_hex:
            self.master_key = bytes.fromhex(master_key_hex)
        elif key_file.exists():
            self.master_key = bytes.fromhex(key_file.read_text().strip())
        else:
            self.master_key = secrets.token_bytes(32)  # 256-bit key
            key_file.write_text(self.master_key.hex())

        # 2. Initialize or load salt
        salt_file = self.vault_dir / "vault_salt.txt"
        if salt:
            self.salt = salt
        elif salt_file.exists():
            self.salt = salt_file.read_text().strip()
        else:
            self.salt = secrets.token_hex(16)
            salt_file.write_text(self.salt)

        self.aesgcm = AESGCM(self.master_key)

    def anonymize_id(self, patient_id: str) -> str:
        """One-way salted hash of patient ID to prevent reverse identification on public ledgers."""
        digest = hashlib.sha256(f"{patient_id}:{self.salt}".encode("utf-8")).digest()
        return "0x" + digest.hex()

    def compute_fingerprint(self, record_data: dict[str, Any]) -> str:
        """
        Computes a deterministic, canonical SHA-256 fingerprint over the patient record.
        Excludes transient UI fields if present.
        """
        clean_record = {
            k: v for k, v in record_data.items()
            if not k.startswith("_") and k not in {"anchored_on_polygon", "polygon_tx_hash", "integrity_status"}
        }
        canonical_json = json.dumps(clean_record, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical_json.encode("utf-8")).digest()
        return "0x" + digest.hex()

    def encrypt_record(self, record_data: dict[str, Any]) -> bytes:
        """Encrypts patient data using AES-256-GCM with a fresh 12-byte nonce."""
        canonical_json = json.dumps(record_data, sort_keys=True, separators=(",", ":"), default=str)
        raw_bytes = canonical_json.encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self.aesgcm.encrypt(nonce, raw_bytes, None)
        return nonce + ciphertext

    def decrypt_record(self, payload: bytes) -> dict[str, Any]:
        """Decrypts AES-256-GCM payload."""
        if len(payload) < 28:
            raise ValueError("Invalid encrypted payload (too short)")
        nonce = payload[:12]
        ciphertext = payload[12:]
        decrypted_bytes = self.aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(decrypted_bytes.decode("utf-8"))

    def store_encrypted_record(self, patient_id: str, record_data: dict[str, Any]) -> tuple[str, str]:
        """
        Encrypts and writes patient record to disk.
        Returns (patient_hash, data_fingerprint).
        """
        patient_hash = self.anonymize_id(patient_id)
        data_fingerprint = self.compute_fingerprint(record_data)
        encrypted_bytes = self.encrypt_record(record_data)

        # Store encrypted payload
        file_path = self.records_dir / f"{patient_id}.enc"
        file_path.write_bytes(encrypted_bytes)

        # Store metadata manifest
        meta_path = self.records_dir / f"{patient_id}.meta.json"
        meta = {
            "patient_id": patient_id,
            "patient_hash": patient_hash,
            "data_fingerprint": data_fingerprint,
            "size_bytes": len(encrypted_bytes),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return patient_hash, data_fingerprint

    def load_decrypted_record(self, patient_id: str) -> dict[str, Any] | None:
        """Loads and decrypts patient record from the vault."""
        file_path = self.records_dir / f"{patient_id}.enc"
        if not file_path.exists():
            return None
        return self.decrypt_record(file_path.read_bytes())
