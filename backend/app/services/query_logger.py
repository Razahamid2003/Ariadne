"""Local query log.

Purpose
-------
Records questions locally (to a JSONL file) so usage can be reviewed and cleared,
with no data leaving the machine.

What it does
------------
Appends query events, returns recent entries, and clears the log on request.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


class QueryLogger:
    """Append local query events to storage/logs/query_log.jsonl."""

    def __init__(self, logs_dir: str | Path):
        self.logs_dir = Path(logs_dir)
        self.log_path = self.logs_dir / "query_log.jsonl"
        self._lock = RLock()

    def log(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def clear(self) -> dict[str, Any]:
        """Delete the local query log file and return a small summary."""

        with self._lock:
            existed = self.log_path.exists()
            if existed:
                try:
                    line_count = len(self.log_path.read_text(encoding="utf-8").splitlines())
                except Exception:
                    line_count = 0
                self.log_path.unlink(missing_ok=True)
            else:
                line_count = 0
        return {"cleared": True, "existed": existed, "records_removed": line_count}

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        limit = max(1, int(limit))
        with self._lock:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()

        records: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(records))
