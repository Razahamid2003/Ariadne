"""SQLite metadata store.

Purpose
-------
Persists ingested documents, chunks, and ingestion-run summaries to a local SQLite
database, with no external database server.

What it does
------------
Creates the schema, upserts documents and chunks, records ingestion runs, returns
counts, and lists chunks for index building.

Flow
----
Ingestion writes documents and chunks here; the index builders read chunks back
from here to build the keyword and vector indexes. All access uses Python's
built-in SQLite module.
"""

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.core.schemas import ChunkRecord, DocumentRecord


class SQLiteMetadataStore:
    """
    Lightweight SQLite metadata store using Python's built-in sqlite3 module.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        """
        Create database directory and required tables if needed.
        """

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            self._create_documents_table(conn)
            self._create_chunks_table(conn)
            self._create_ingestion_runs_table(conn)
            conn.commit()

    def upsert_documents(self, documents: list[DocumentRecord]) -> None:
        """
        Insert or replace document records.
        """

        if not documents:
            return

        self.initialize()

        rows = [
            (
                document.document_id,
                document.source_file,
                document.source_system,
                document.record_type,
                document.title,
                document.sensitivity,
                document.created_at.isoformat(),
                self._json(document.metadata),
            )
            for document in documents
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO documents (
                    document_id,
                    source_file,
                    source_system,
                    record_type,
                    title,
                    sensitivity,
                    created_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """
        Insert or replace chunk records.
        """

        if not chunks:
            return

        self.initialize()

        rows = [
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.text,
                chunk.source_file,
                chunk.source_system,
                chunk.record_type,
                chunk.title,
                chunk.chunk_index,
                chunk.sensitivity,
                chunk.citation_label(),
                self._json(chunk.metadata),
            )
            for chunk in chunks
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id,
                    document_id,
                    text,
                    source_file,
                    source_system,
                    record_type,
                    title,
                    chunk_index,
                    sensitivity,
                    citation_label,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()

    def save_ingestion_run(self, report: Any) -> str:
        """
        Save one ingestion-run summary.

        Args:
            report:
                Dataclass or dictionary-like report object.

        Returns:
            Generated ingestion run ID.
        """

        self.initialize()

        run_id = self._new_run_id()
        payload = self._to_dict(report)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id,
                    started_at,
                    input_dir,
                    output_dir,
                    files_seen,
                    files_processed,
                    files_skipped,
                    documents_created,
                    chunks_created,
                    errors_json,
                    report_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(),
                    payload.get("input_dir", ""),
                    payload.get("output_dir", ""),
                    int(payload.get("files_seen", 0)),
                    int(payload.get("files_processed", 0)),
                    int(payload.get("files_skipped", 0)),
                    int(payload.get("documents_created", 0)),
                    int(payload.get("chunks_created", 0)),
                    self._json(payload.get("errors", [])),
                    self._json(payload),
                ),
            )
            conn.commit()

        return run_id

    def count_documents(self) -> int:
        self.initialize()

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()
            return int(row[0])

    def count_chunks(self) -> int:
        self.initialize()

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()
            return int(row[0])

    def count_ingestion_runs(self) -> int:
        self.initialize()

        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM ingestion_runs;").fetchone()
            return int(row[0])
        
    
    def list_chunks_for_indexing(self) -> list[dict[str, Any]]:
        """
        Return all chunks needed for embedding and vector indexing.

        The returned rows are ordered by chunk_id for deterministic indexing.
        """

        self.initialize()

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    text,
                    source_file,
                    source_system,
                    record_type,
                    title,
                    chunk_index,
                    sensitivity,
                    citation_label,
                    metadata_json
                FROM chunks
                ORDER BY chunk_id ASC;
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def list_recent_chunks(self, limit: int = 5) -> list[dict[str, Any]]:
        """
        Return recent chunks for inspection.
        """

        self.initialize()

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    source_system,
                    source_file,
                    record_type,
                    title,
                    chunk_index,
                    citation_label,
                    substr(text, 1, 250) AS text_preview
                FROM chunks
                ORDER BY rowid DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        """
        Open a SQLite connection with foreign keys enabled.
        """

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @staticmethod
    def _create_documents_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_file TEXT NOT NULL,
                source_system TEXT NOT NULL,
                record_type TEXT NOT NULL,
                title TEXT,
                sensitivity TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_source_system
            ON documents(source_system);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_record_type
            ON documents(record_type);
            """
        )

    @staticmethod
    def _create_chunks_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                source_file TEXT NOT NULL,
                source_system TEXT NOT NULL,
                record_type TEXT NOT NULL,
                title TEXT,
                chunk_index INTEGER NOT NULL,
                sensitivity TEXT NOT NULL,
                citation_label TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
                    ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_source_system
            ON chunks(source_system);
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_record_type
            ON chunks(record_type);
            """
        )

    @staticmethod
    def _create_ingestion_runs_table(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                input_dir TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                files_seen INTEGER NOT NULL,
                files_processed INTEGER NOT NULL,
                files_skipped INTEGER NOT NULL,
                documents_created INTEGER NOT NULL,
                chunks_created INTEGER NOT NULL,
                errors_json TEXT NOT NULL,
                report_json TEXT NOT NULL
            );
            """
        )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value

        if is_dataclass(value):
            return asdict(value)

        if hasattr(value, "__dict__"):
            return dict(value.__dict__)

        raise TypeError(f"Cannot convert object to dict: {type(value)}")

    @staticmethod
    def _new_run_id() -> str:
        return "run-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")