"""
Append-only Audit Log for Regulatory and Compliance Tracking.
Tracks: who, when, endpoint, case_id, model_version, input file hashes.
Strictly zero raw patient values in logs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bio_nexus.serving.config import settings

logger = logging.getLogger("bionexus.audit")


class AuditLogger:
    def __init__(self, log_path: Path | None = None):
        self.log_path = log_path or settings.audit_log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_event(
        self,
        who: str,
        endpoint: str,
        action: str,
        case_id: str | None = None,
        model_version: str | None = None,
        input_file_hashes: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "who": who,
            "endpoint": endpoint,
            "action": action,
            "case_id": case_id or "N/A",
            "model_version": model_version or settings.model_version,
            "input_file_hashes": input_file_hashes or {},
            "details": details or {},
        }

        line = json.dumps(entry) + "\n"
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error("Failed to write audit log: %s", e)


audit_logger = AuditLogger()
