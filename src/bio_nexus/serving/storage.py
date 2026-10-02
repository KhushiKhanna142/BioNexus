"""
Storage layer for Case Intake, Artifacts, and Metadata.
Abstract CaseStorage interface with LocalCaseStorage (default) and extensible S3 backend.
"""

from __future__ import annotations

import abc
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bio_nexus.serving.config import settings


class CaseRecord(BaseModel):
    case_id: str
    patient_ref: str = Field(description="Pseudonymous patient identifier")
    specimen_type: str = "primary_tumor"
    status: str = "intake_qc_passed"  # intake_qc_passed, predicted, failed
    created_at: str
    updated_at: str
    modalities_available: list[str]
    input_file_hashes: dict[str, str] = Field(default_factory=dict)
    qc_summary: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    audit_notes: list[str] = Field(default_factory=list)


def validate_patient_ref(ref: str) -> bool:
    """Privacy safeguard: validate patient_ref is pseudonymous (no spaces, names, PII)."""
    if not ref or len(ref) > 64:
        return False
    # Allow alphanumeric, hyphens, underscores, dots (e.g. TCGA-AR-A1AR or CASE_001)
    return bool(re.match(r"^[A-Za-z0-9_.\-]+$", ref))


class CaseStorage(abc.ABC):
    @abc.abstractmethod
    def save_case(
        self,
        case_id: str,
        patient_ref: str,
        specimen_type: str,
        modalities: list[str],
        qc_summary: dict[str, Any],
        raw_files: dict[str, bytes],
        aligned_vectors: dict[str, dict[str, float]],
    ) -> CaseRecord:
        pass

    @abc.abstractmethod
    def get_case(self, case_id: str) -> CaseRecord | None:
        pass

    @abc.abstractmethod
    def get_aligned_vectors(self, case_id: str) -> dict[str, dict[str, float]] | None:
        pass

    @abc.abstractmethod
    def list_cases(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[CaseRecord]:
        pass

    @abc.abstractmethod
    def update_case_prediction(self, case_id: str, prediction: dict[str, Any]) -> CaseRecord | None:
        pass

    @abc.abstractmethod
    def delete_case(self, case_id: str) -> bool:
        pass


class LocalCaseStorage(CaseStorage):
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or settings.storage_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _case_dir(self, case_id: str) -> Path:
        return self.base_dir / case_id

    def save_case(
        self,
        case_id: str,
        patient_ref: str,
        specimen_type: str,
        modalities: list[str],
        qc_summary: dict[str, Any],
        raw_files: dict[str, bytes],
        aligned_vectors: dict[str, dict[str, float]],
    ) -> CaseRecord:
        cdir = self._case_dir(case_id)
        cdir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()
        hashes: dict[str, str] = {}

        # Save raw files
        for fname, data in raw_files.items():
            fpath = cdir / fname
            fpath.write_bytes(data)
            hashes[fname] = hashlib.sha256(data).hexdigest()

        # Save aligned vectors
        vec_path = cdir / "aligned_vectors.json"
        vec_path.write_text(json.dumps(aligned_vectors, indent=2))

        record = CaseRecord(
            case_id=case_id,
            patient_ref=patient_ref,
            specimen_type=specimen_type,
            status="intake_qc_passed",
            created_at=now,
            updated_at=now,
            modalities_available=modalities,
            input_file_hashes=hashes,
            qc_summary=qc_summary,
            prediction=None,
            audit_notes=["Case created and QC validated"],
        )

        meta_path = cdir / "metadata.json"
        meta_path.write_text(record.model_dump_json(indent=2))
        return record

    def get_case(self, case_id: str) -> CaseRecord | None:
        meta_path = self._case_dir(case_id) / "metadata.json"
        if not meta_path.exists():
            return None
        try:
            return CaseRecord.model_validate_json(meta_path.read_text())
        except Exception:
            return None

    def get_aligned_vectors(self, case_id: str) -> dict[str, dict[str, float]] | None:
        vec_path = self._case_dir(case_id) / "aligned_vectors.json"
        if not vec_path.exists():
            return None
        try:
            return json.loads(vec_path.read_text())
        except Exception:
            return None

    def list_cases(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[CaseRecord]:
        records: list[CaseRecord] = []
        for cdir in self.base_dir.iterdir():
            if cdir.is_dir():
                rec = self.get_case(cdir.name)
                if rec:
                    if status is None or rec.status == status:
                        records.append(rec)

        # Sort newest first
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def update_case_prediction(self, case_id: str, prediction: dict[str, Any]) -> CaseRecord | None:
        rec = self.get_case(case_id)
        if not rec:
            return None
        rec.prediction = prediction
        rec.status = "predicted"
        rec.updated_at = datetime.now(timezone.utc).isoformat()
        rec.audit_notes.append("Prediction completed")

        meta_path = self._case_dir(case_id) / "metadata.json"
        meta_path.write_text(rec.model_dump_json(indent=2))
        return rec

    def delete_case(self, case_id: str) -> bool:
        cdir = self._case_dir(case_id)
        if not cdir.exists():
            return False
        import shutil
        shutil.rmtree(cdir, ignore_errors=True)
        return True


# Global storage instance
storage: CaseStorage = LocalCaseStorage()
