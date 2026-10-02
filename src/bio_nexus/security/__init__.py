"""
bio_nexus.security
-------------------
Cryptographic security, patient privacy vault, and Polygon blockchain anchoring.
"""

from bio_nexus.security.cohort_manager import CohortBatchManager, get_cohort_manager
from bio_nexus.security.merkle_tree import MerkleTree, compute_patient_leaf
from bio_nexus.security.patient_vault import PatientVault
from bio_nexus.security.polygon_service import PolygonService

_VAULT_INSTANCE: PatientVault | None = None
_POLYGON_INSTANCE: PolygonService | None = None


def get_patient_vault() -> PatientVault:
    global _VAULT_INSTANCE
    if _VAULT_INSTANCE is None:
        _VAULT_INSTANCE = PatientVault()
    return _VAULT_INSTANCE


def get_polygon_service() -> PolygonService:
    global _POLYGON_INSTANCE
    if _POLYGON_INSTANCE is None:
        _POLYGON_INSTANCE = PolygonService()
    return _POLYGON_INSTANCE


__all__ = [
    "PatientVault",
    "PolygonService",
    "MerkleTree",
    "compute_patient_leaf",
    "CohortBatchManager",
    "get_patient_vault",
    "get_polygon_service",
    "get_cohort_manager",
]

