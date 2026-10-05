"""
Configuration for BioNexus Serving Layer using pydantic-settings.
Supports environment variables prefixed with BIONEXUS_.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BIONEXUS_",
        env_file=".env",
        extra="ignore",
    )

    # Server & Environment
    env: str = Field(default="development", description="Environment: development, staging, production")
    dev_mode: bool = Field(default=True, description="When true, bypasses strict auth for dev convenience")
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: Union[list[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:4173",
    ]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except Exception:
                    pass
            return [x.strip() for x in s.split(",") if x.strip()]
        elif isinstance(v, (list, tuple)):
            return [str(x) for x in v]
        return v

    # Storage
    storage_backend: str = Field(default="local", description="local or s3")
    storage_dir: Path = Field(default=Path("data/cases"), description="Directory for storing case files")
    s3_bucket: str | None = None
    encryption_enabled: bool = False

    # Security & Auth
    # Pre-configured keys or hash verification. Keys format: {"key_string": "role"}
    api_keys: dict[str, str] = Field(
        default_factory=lambda: {
            "dev-admin-key": "admin",
            "dev-operator-key": "operator",
            "dev-viewer-key": "viewer",
        }
    )
    rate_limit_per_minute: int = 120

    # Quality Control Thresholds
    qc_coverage_reject_threshold: float = Field(default=0.50, description="Reject if mapped features < 50%")
    qc_coverage_warn_threshold: float = Field(default=0.80, description="Warn if mapped features < 80%")
    qc_max_acceptable_missing: float = Field(default=0.30, description="Max missing values before reject")

    # ML Inference & Thresholds
    confidence_tier_high: float = Field(default=0.80, description="High confidence threshold")
    confidence_tier_medium: float = Field(default=0.50, description="Medium confidence threshold")
    abstain_confidence_threshold: float = Field(default=0.45, description="Abstain if max calibrated prob < threshold")
    model_version: str = "1.0.0"
    preprocessing_version: str = "1.0.0"

    # Audit & Privacy
    audit_log_path: Path = Field(default=Path("data/audit/audit.log"))
    retention_days: int = 90


# Singleton instance
settings = ServingSettings()
