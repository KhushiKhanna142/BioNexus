"""
Quality Control (QC) and Feature Alignment for Case Intake.
Validates input gene/probe IDs, version suffixes, scale, coverage, and duplicate IDs against model features.
Supports configurable thresholds from configs/qc.yaml, auto-normalization, and plain-language fix hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd
import yaml

from bio_nexus.serving.config import settings

QCStatus = Literal["PASSED", "WARNING", "FAILED"]


@dataclass
class ModalityQCResult:
    modality: str
    status: QCStatus
    total_expected: int
    mapped_count: int
    coverage_pct: float
    missing_pct: float
    value_min: float
    value_max: float
    value_mean: float
    detected_scale: str  # "raw_counts", "log2", "beta_values", "unknown"
    likely_raw_counts: bool
    duplicate_ids_count: int = 0
    duplicate_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fix_hints: list[str] = field(default_factory=list)
    auto_normalized_applied: bool = False
    aligned_vector: dict[str, float] = field(default_factory=dict)


@dataclass
class CaseQCReport:
    overall_status: QCStatus
    modalities_checked: list[str]
    results: dict[str, ModalityQCResult]
    can_proceed: bool
    requires_caution: bool
    ready_modalities_count: int
    summary_message: str
    auto_normalized_modalities: list[str] = field(default_factory=list)


class FeatureDictionaryManager:
    """Manages expected feature lists, symbol/ID aliases, duplicate tracking, and scale detection."""

    def __init__(self, data_dir: Path | None = None, config_path: Path | None = None):
        base_dir = Path(__file__).resolve().parents[3]
        if data_dir is None:
            data_dir = base_dir / "data" / "serving"
        if config_path is None:
            config_path = base_dir / "configs" / "qc.yaml"

        self.data_dir = data_dir
        self.config_path = config_path

        self.expected_features: dict[str, list[str]] = {}
        self.feature_symbols: dict[str, dict[str, str]] = {}  # id -> symbol
        self.symbol_to_id: dict[str, dict[str, str]] = {}    # symbol -> id
        self.clean_id_to_full_id: dict[str, dict[str, str]] = {}

        # Default configuration thresholds
        self.coverage_reject_threshold = settings.qc_coverage_reject_threshold  # 0.50
        self.coverage_warn_threshold = settings.qc_coverage_warn_threshold      # 0.80
        self.scale_cfg = {
            "rna": {"raw_count_max": 30.0, "raw_count_mean": 50.0},
            "mirna": {"raw_count_max": 35.0},
            "meth": {"min_beta": -0.05, "max_beta": 1.05},
        }

        self._load_config()
        self._load_dictionaries()

    def _load_config(self) -> None:
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if cfg and "qc" in cfg:
                    q = cfg["qc"]
                    self.coverage_reject_threshold = float(q.get("coverage_reject_threshold", self.coverage_reject_threshold))
                    self.coverage_warn_threshold = float(q.get("coverage_warn_threshold", self.coverage_warn_threshold))
                    if "scale_detection" in q:
                        self.scale_cfg.update(q["scale_detection"])
            except Exception as e:
                pass

    def _load_dictionaries(self) -> None:
        for mod in ["rna", "mirna", "meth"]:
            feat_file = self.data_dir / f"{mod}_features.csv"
            if not feat_file.exists():
                continue
            df = pd.read_csv(feat_file)
            ids = df["feature_id"].astype(str).tolist()
            symbols = df["symbol"].astype(str).tolist() if "symbol" in df.columns else ids

            self.expected_features[mod] = ids
            self.feature_symbols[mod] = dict(zip(ids, symbols, strict=False))

            # Symbol to id mapping (case-insensitive keys for robustness)
            sym_map = {}
            clean_map = {}
            for full_id, sym in zip(ids, symbols, strict=False):
                sym_clean = sym.strip().upper()
                if sym_clean:
                    sym_map[sym_clean] = full_id
                # Strip Ensembl version suffixes (e.g. ENSG00000123456.12 -> ENSG00000123456)
                clean_id = re.sub(r"\.\d+$", "", full_id.strip())
                clean_map[clean_id] = full_id
                clean_map[full_id] = full_id

            self.symbol_to_id[mod] = sym_map
            self.clean_id_to_full_id[mod] = clean_map

    def align_and_qc(
        self,
        modality: str,
        input_data: dict[str, float] | pd.DataFrame | list[dict],
        auto_normalize: bool = False,
    ) -> ModalityQCResult:
        # Standardize modality alias (e.g. methylation -> meth)
        mod_key = "meth" if modality in ["meth", "methylation"] else modality

        if mod_key not in self.expected_features:
            return ModalityQCResult(
                modality=modality,
                status="FAILED",
                total_expected=0,
                mapped_count=0,
                coverage_pct=0.0,
                missing_pct=1.0,
                value_min=0.0,
                value_max=0.0,
                value_mean=0.0,
                detected_scale="unknown",
                likely_raw_counts=False,
                errors=[f"Unsupported or unconfigured modality: {modality}"],
                fix_hints=["Upload RNA-seq, miRNA, or DNA Methylation data."],
            )

        expected = self.expected_features[mod_key]
        total_exp = len(expected)
        sym_map = self.symbol_to_id.get(mod_key, {})
        clean_map = self.clean_id_to_full_id.get(mod_key, {})

        # Parse input into list of tuples to track duplicates: [(raw_key, val)]
        raw_entries: list[tuple[str, float]] = []
        if isinstance(input_data, pd.DataFrame):
            if input_data.empty:
                raw_entries = []
            elif input_data.shape[1] >= 2:
                col0, col1 = input_data.columns[0], input_data.columns[1]
                for _, r in input_data.iterrows():
                    try:
                        raw_entries.append((str(r[col0]), float(r[col1])))
                    except (ValueError, TypeError):
                        continue
            elif input_data.shape[0] == 1:
                for c in input_data.columns:
                    val = input_data.iloc[0][c]
                    if pd.notna(val):
                        try:
                            raw_entries.append((str(c), float(val)))
                        except (ValueError, TypeError):
                            continue
        elif isinstance(input_data, dict):
            for k, v in input_data.items():
                if v is not None and pd.notna(v):
                    try:
                        raw_entries.append((str(k), float(v)))
                    except (ValueError, TypeError):
                        continue
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict):
                    k = item.get("id") or item.get("gene") or item.get("feature_id") or list(item.keys())[0]
                    v = item.get("value") if "value" in item else list(item.values())[0]
                    try:
                        raw_entries.append((str(k), float(v)))
                    except (ValueError, TypeError):
                        continue

        # Check for duplicated IDs
        seen_keys: set[str] = set()
        duplicate_keys: list[str] = []
        deduped_dict: dict[str, float] = {}

        for raw_k, val in raw_entries:
            k_strip = raw_k.strip()
            if k_strip in seen_keys:
                duplicate_keys.append(k_strip)
            else:
                seen_keys.add(k_strip)
                deduped_dict[k_strip] = val

        # Match against expected features
        aligned: dict[str, float] = {}
        matched_values: list[float] = []

        for raw_k, val in deduped_dict.items():
            target_id = None
            if raw_k in clean_map:
                target_id = clean_map[raw_k]
            else:
                base_k = re.sub(r"\.\d+$", "", raw_k)
                if base_k in clean_map:
                    target_id = clean_map[base_k]
                elif raw_k.upper() in sym_map:
                    target_id = sym_map[raw_k.upper()]

            if target_id and target_id in expected and target_id not in aligned:
                aligned[target_id] = val
                matched_values.append(val)

        mapped_count = len(aligned)
        cov_pct = mapped_count / total_exp if total_exp > 0 else 0.0
        missing_pct = 1.0 - cov_pct

        warnings: list[str] = []
        errors: list[str] = []
        fix_hints: list[str] = []

        # Duplicate ID reporting
        dup_count = len(duplicate_keys)
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicated row identifiers. Only the first occurrence was retained.")
            fix_hints.append(f"Remove {dup_count} duplicate rows (e.g. {duplicate_keys[:3]}) from your input file.")

        if matched_values:
            val_arr = np.array(matched_values, dtype=np.float32)
            v_min = float(np.min(val_arr))
            v_max = float(np.max(val_arr))
            v_mean = float(np.mean(val_arr))
        else:
            v_min, v_max, v_mean = 0.0, 0.0, 0.0

        # Scale detection heuristics
        detected_scale = "unknown"
        likely_raw = False

        if mod_key == "rna":
            rna_cfg = self.scale_cfg.get("rna", {})
            raw_max_thresh = rna_cfg.get("raw_count_max", 30.0)
            raw_mean_thresh = rna_cfg.get("raw_count_mean", 50.0)
            if v_max > raw_max_thresh or v_mean > raw_mean_thresh:
                likely_raw = True
                detected_scale = "raw_counts"
                warnings.append(
                    f"Likely raw count scale detected (max={v_max:.1f}, mean={v_mean:.1f}). "
                    "BioNexus expects log2(TPM + 1) normalized values."
                )
                fix_hints.append(
                    "Check the 'Auto-normalize (log2 TPM+1)' checkbox or pre-normalize values to log2(TPM+1)."
                )
            else:
                detected_scale = "log2"
        elif mod_key == "mirna":
            mirna_cfg = self.scale_cfg.get("mirna", {})
            if v_max > mirna_cfg.get("raw_count_max", 35.0):
                likely_raw = True
                detected_scale = "raw_counts"
                warnings.append(
                    f"Unusually high miRNA value detected (max={v_max:.1f}). Expected log2(RPM + 1)."
                )
                fix_hints.append("Pre-normalize miRNA expression to log2(RPM+1).")
            else:
                detected_scale = "log2"
        elif mod_key == "meth":
            meth_cfg = self.scale_cfg.get("meth", {})
            min_beta = meth_cfg.get("min_beta", -0.05)
            max_beta = meth_cfg.get("max_beta", 1.05)
            if v_min < min_beta or v_max > max_beta:
                detected_scale = "unknown"
                warnings.append(
                    f"Methylation beta values outside expected [0.0, 1.0] range (min={v_min:.2f}, max={v_max:.2f})."
                )
                fix_hints.append("Verify methylation values are beta values (0.0 to 1.0), not M-values.")
            else:
                detected_scale = "beta_values"

        # Explicit Auto-normalization
        auto_norm_applied = False
        if auto_normalize and likely_raw and mod_key in ["rna", "mirna"]:
            aligned = {k: float(np.log2(max(0.0, v) + 1.0)) for k, v in aligned.items()}
            auto_norm_applied = True
            detected_scale = "log2_normalized (auto-applied)"
            warnings.append("Auto-normalization log2(x + 1) was explicitly applied.")
            if aligned:
                v_max = max(aligned.values())
                v_min = min(aligned.values())
                v_mean = float(np.mean(list(aligned.values())))

        if not raw_entries:
            errors.append("File is empty or contains no valid rows.")
            fix_hints.append("File is empty or contains no valid (id, value) rows.")

        # Coverage QC thresholds
        if cov_pct < self.coverage_reject_threshold:
            status: QCStatus = "FAILED"
            errors.append(
                f"Feature coverage ({cov_pct * 100:.1f}%) is below minimum required threshold "
                f"({self.coverage_reject_threshold * 100:.0f}%). "
                f"Only {mapped_count} of {total_exp} expected features matched."
            )
            fix_hints.append(
                f"Ensure gene symbols or Ensembl IDs match the model feature dictionary ({total_exp} features)."
            )
        elif cov_pct < self.coverage_warn_threshold:
            status = "WARNING"
            warnings.append(
                f"Feature coverage ({cov_pct * 100:.1f}%) is sub-optimal "
                f"(recommended: >={self.coverage_warn_threshold * 100:.0f}%). "
                f"{total_exp - mapped_count} features missing."
            )
            fix_hints.append(
                "Submitting with partial coverage may degrade prediction certainty. Escalation may be needed."
            )
        else:
            status = "PASSED"

        return ModalityQCResult(
            modality=modality,
            status=status,
            total_expected=total_exp,
            mapped_count=mapped_count,
            coverage_pct=round(cov_pct, 4),
            missing_pct=round(missing_pct, 4),
            value_min=round(v_min, 4),
            value_max=round(v_max, 4),
            value_mean=round(v_mean, 4),
            detected_scale=detected_scale,
            likely_raw_counts=likely_raw,
            duplicate_ids_count=dup_count,
            duplicate_ids=duplicate_keys[:10],
            warnings=warnings,
            errors=errors,
            fix_hints=fix_hints,
            auto_normalized_applied=auto_norm_applied,
            aligned_vector=aligned,
        )

    def run_case_qc(
        self,
        modality_inputs: dict[str, Any],
        auto_normalize: bool = False,
    ) -> CaseQCReport:
        results: dict[str, ModalityQCResult] = {}
        all_passed = True
        has_warnings = False
        auto_normalized_mods: list[str] = []

        if not modality_inputs:
            return CaseQCReport(
                overall_status="FAILED",
                modalities_checked=[],
                results={},
                can_proceed=False,
                requires_caution=False,
                ready_modalities_count=0,
                summary_message="No modalities supplied. At least one omics modality is required.",
            )

        ready_count = 0
        for mod, inp in modality_inputs.items():
            res = self.align_and_qc(mod, inp, auto_normalize=auto_normalize)
            results[mod] = res
            if res.auto_normalized_applied:
                auto_normalized_mods.append(mod)

            if res.status == "FAILED":
                all_passed = False
            elif res.status == "WARNING":
                has_warnings = True
                ready_count += 1
            else:
                ready_count += 1

        total_checked = len(results)

        if not all_passed:
            overall: QCStatus = "FAILED"
            can_proceed = False
            requires_caution = False
            summary = f"Case QC failed: {ready_count} of {total_checked} data types ready. One or more failed minimum coverage."
        elif has_warnings:
            overall = "WARNING"
            can_proceed = True
            requires_caution = True
            summary = f"Ready to run with {ready_count} of 3 data types (warnings detected)."
        else:
            overall = "PASSED"
            can_proceed = True
            requires_caution = False
            summary = f"Ready to run with {ready_count} of 3 data types."

        return CaseQCReport(
            overall_status=overall,
            modalities_checked=list(results.keys()),
            results=results,
            can_proceed=can_proceed,
            requires_caution=requires_caution,
            ready_modalities_count=ready_count,
            summary_message=summary,
            auto_normalized_modalities=auto_normalized_mods,
        )

    align_and_qc_modality = align_and_qc
    check_modality = align_and_qc


# Global feature manager instance
qc_manager = FeatureDictionaryManager()
QCManager = FeatureDictionaryManager
