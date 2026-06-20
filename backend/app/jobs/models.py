"""Job record model.

Purpose
-------
Defines the record that describes one background job (its id, state, timing, and
result) and serializes it for the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    """One background job tracked by the local in-memory job manager."""

    job_id: str
    name: str
    status: str = "queued"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None

    def to_dict(self, include_result: bool = True) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        if include_result:
            payload["result"] = self.result
        return payload
