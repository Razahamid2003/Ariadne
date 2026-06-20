"""File lifecycle registry.

Purpose
-------
Tracks the lifecycle of every input file (seen, ingested, missing) separately from
the document and chunk data, so the system knows what to add, update, or remove.

What it does
------------
Stores one row per known input file in SQLite and offers operations to record that
a file was seen or ingested, mark files missing, list everything tracked, and
summarize counts.

Flow
----
During a scan each present file is upserted as "seen"; files no longer present are
marked "missing"; after successful processing a file is marked "ingested". The
registry is the source of truth for incremental decisions.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from backend.app.intake.file_fingerprint import FileFingerprint


@dataclass(frozen=True)
class RegistryRow:
    """
    One row from file_registry.
    """

    normalized_path: str
    file_path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_time_ns: int
    modified_time_iso: str
    sha256: str
    signature: str
    status: str
    document_id: str | None
    chunks_count: int
    last_seen_at: str | None
    last_ingested_at: str | None
    last_error: str | None


class FileRegistry:
    """
    Persistence layer for file lifecycle tracking.
    """

    def __init__(self, metadata_db_path: str | Path):
        self.metadata_db_path = Path(metadata_db_path)

    def _connect(self) -> sqlite3.Connection:
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """
        Create the registry table if missing.
        """

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_registry (
                    normalized_path TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_time_ns INTEGER NOT NULL,
                    modified_time_iso TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status TEXT NOT NULL,
                    document_id TEXT,
                    chunks_count INTEGER NOT NULL DEFAULT 0,
                    last_seen_at TEXT,
                    last_ingested_at TEXT,
                    last_error TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_registry_status
                ON file_registry(status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_registry_sha256
                ON file_registry(sha256);
                """
            )
            conn.commit()

    def get(self, normalized_path: str) -> RegistryRow | None:
        """
        Fetch one file registry row.
        """

        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM file_registry WHERE normalized_path = ?;",
                (normalized_path,),
            ).fetchone()
        return self._row_to_model(row) if row else None

    def list_all(self) -> list[RegistryRow]:
        """
        Return all tracked files.
        """

        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM file_registry ORDER BY normalized_path ASC;"
            ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def known_paths(self) -> set[str]:
        """
        Return all normalized paths known to the registry.
        """

        return {row.normalized_path for row in self.list_all()}

    def upsert_seen(
        self,
        fingerprint: FileFingerprint,
        status: str,
        document_id: str | None = None,
        chunks_count: int = 0,
        error: str | None = None,
    ) -> None:
        """
        Insert/update a row for a file currently seen on disk.
        """

        now = utc_now()
        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_registry (
                    normalized_path,
                    file_path,
                    file_name,
                    extension,
                    size_bytes,
                    modified_time_ns,
                    modified_time_iso,
                    sha256,
                    signature,
                    status,
                    document_id,
                    chunks_count,
                    last_seen_at,
                    last_ingested_at,
                    last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(normalized_path) DO UPDATE SET
                    file_path = excluded.file_path,
                    file_name = excluded.file_name,
                    extension = excluded.extension,
                    size_bytes = excluded.size_bytes,
                    modified_time_ns = excluded.modified_time_ns,
                    modified_time_iso = excluded.modified_time_iso,
                    sha256 = excluded.sha256,
                    signature = excluded.signature,
                    status = excluded.status,
                    document_id = COALESCE(excluded.document_id, file_registry.document_id),
                    chunks_count = excluded.chunks_count,
                    last_seen_at = excluded.last_seen_at,
                    last_ingested_at = excluded.last_ingested_at,
                    last_error = excluded.last_error;
                """,
                (
                    fingerprint.normalized_path,
                    fingerprint.file_path,
                    fingerprint.file_name,
                    fingerprint.extension,
                    fingerprint.size_bytes,
                    fingerprint.modified_time_ns,
                    fingerprint.modified_time_iso,
                    fingerprint.sha256,
                    fingerprint.signature,
                    status,
                    document_id,
                    chunks_count,
                    now,
                    now if status == "ingested" else None,
                    error,
                ),
            )
            conn.commit()

    def mark_missing(self, normalized_path: str) -> None:
        """
        Mark a previously tracked path as missing/deleted.
        """

        self.initialize()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE file_registry
                SET status = 'missing', last_seen_at = ?, last_error = NULL
                WHERE normalized_path = ?;
                """,
                (utc_now(), normalized_path),
            )
            conn.commit()

    def mark_ingested(
        self,
        fingerprint: FileFingerprint,
        document_id: str | None,
        chunks_count: int,
        error: str | None = None,
    ) -> None:
        """
        Mark a file as successfully ingested.
        """

        self.upsert_seen(
            fingerprint=fingerprint,
            status="ingested" if error is None else "failed",
            document_id=document_id,
            chunks_count=chunks_count,
            error=error,
        )

    def summary(self) -> dict:
        """
        Return status counts for dashboards/CLI.
        """

        self.initialize()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM file_registry
                GROUP BY status
                ORDER BY status ASC;
                """
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) FROM file_registry;").fetchone()[0]
        return {
            "exists": True,
            "total_files_tracked": total,
            "status_counts": {row["status"]: row["count"] for row in rows},
        }

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> RegistryRow:
        return RegistryRow(
            normalized_path=row["normalized_path"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            modified_time_ns=row["modified_time_ns"],
            modified_time_iso=row["modified_time_iso"],
            sha256=row["sha256"],
            signature=row["signature"],
            status=row["status"],
            document_id=row["document_id"],
            chunks_count=row["chunks_count"],
            last_seen_at=row["last_seen_at"],
            last_ingested_at=row["last_ingested_at"],
            last_error=row["last_error"],
        )


def utc_now() -> str:
    """
    UTC timestamp for registry rows.
    """

    return datetime.now(timezone.utc).isoformat()
