"""Incremental ingestion manager.

Purpose
-------
Runs ingestion efficiently by processing only files that are new or changed and
cleaning up records for files that were deleted.

What it does
------------
Scans the input directory, fingerprints each file, compares against the registry
to build a change plan, ingests only the selected files, and removes records for
missing files.

Flow
----
``plan()`` produces the add/update/remove set; ``ingest()`` processes the selected
files through the standard pipeline, updates the registry, and returns a report.
Deleted files have their document and chunk records removed so the index stays in
sync.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from backend.app.ingestion.pipeline import run_ingestion
from backend.app.ingestion.registry import LoaderRegistry
from backend.app.intake.file_fingerprint import (
    FileFingerprint,
    fingerprint_file,
    normalize_relative_path,
    normalize_stored_path,
)
from backend.app.intake.file_registry import FileRegistry


@dataclass
class ChangePlan:
    """
    Result of scanning input files before ingestion.
    """

    input_dir: str
    supported_extensions: list[str]
    files_seen: int = 0
    files_supported: int = 0
    files_new: list[FileFingerprint] = field(default_factory=list)
    files_changed: list[FileFingerprint] = field(default_factory=list)
    files_unchanged: list[FileFingerprint] = field(default_factory=list)
    files_missing: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)

    def selected_for_ingestion(self, force: bool = False) -> list[FileFingerprint]:
        """
        Files that should be ingested.
        """

        if force:
            return [*self.files_new, *self.files_changed, *self.files_unchanged]
        return [*self.files_new, *self.files_changed]

    def to_dict(self) -> dict:
        return {
            "input_dir": self.input_dir,
            "supported_extensions": self.supported_extensions,
            "files_seen": self.files_seen,
            "files_supported": self.files_supported,
            "files_new": len(self.files_new),
            "files_changed": len(self.files_changed),
            "files_unchanged": len(self.files_unchanged),
            "files_missing": len(self.files_missing),
            "files_skipped": len(self.files_skipped),
            "new_files": [item.normalized_path for item in self.files_new],
            "changed_files": [item.normalized_path for item in self.files_changed],
            "missing_files": self.files_missing,
            "skipped_files": self.files_skipped[:50],
        }


@dataclass
class IncrementalIngestionReport:
    """
    Final report returned by incremental ingestion.
    """

    mode: str
    input_dir: str
    metadata_db: str
    files_seen: int
    files_new: int
    files_changed: int
    files_unchanged: int
    files_missing: int
    files_selected_for_ingestion: int
    files_removed_from_index: int
    documents_created: int
    chunks_created: int
    errors: list[str]
    ingestion_report: dict | None = None
    index_status: str = "stale"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "input_dir": self.input_dir,
            "metadata_db": self.metadata_db,
            "files_seen": self.files_seen,
            "files_new": self.files_new,
            "files_changed": self.files_changed,
            "files_unchanged": self.files_unchanged,
            "files_missing": self.files_missing,
            "files_selected_for_ingestion": self.files_selected_for_ingestion,
            "files_removed_from_index": self.files_removed_from_index,
            "documents_created": self.documents_created,
            "chunks_created": self.chunks_created,
            "index_status": self.index_status,
            "errors": self.errors,
            "ingestion_report": self.ingestion_report,
        }


class IncrementalIngestionManager:
    """
    Detect file changes and ingest only changed/new files.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        metadata_db_path: str | Path,
        max_chars: int,
        overlap_chars: int,
        vision_config,
        ocr_config,
        work_dir: str | Path = "storage/incremental_work",
        auto_metadata_config=None,
        llm_client=None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_db_path = Path(metadata_db_path)
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.vision_config = vision_config
        self.ocr_config = ocr_config
        self.auto_metadata_config = auto_metadata_config
        self.llm_client = llm_client
        self.work_dir = Path(work_dir)
        self.registry = FileRegistry(metadata_db_path)

    def plan(self) -> ChangePlan:
        """
        Scan the input folder and classify files.
        """

        self.registry.initialize()

        loader_registry = LoaderRegistry(
            vision_config=self.vision_config,
            ocr_config=self.ocr_config,
        )
        supported_extensions = set(loader_registry.supported_extensions())

        plan = ChangePlan(
            input_dir=str(self.input_dir),
            supported_extensions=sorted(supported_extensions),
        )

        seen_supported_paths: set[str] = set()

        for path in sorted(self.input_dir.rglob("*")):
            if not path.is_file():
                continue
            if is_runtime_or_hidden_file(path):
                continue

            plan.files_seen += 1
            extension = path.suffix.lower()

            if extension not in supported_extensions:
                plan.files_skipped.append(str(path))
                continue

            plan.files_supported += 1
            fp = fingerprint_file(path, self.input_dir)
            seen_supported_paths.add(fp.normalized_path)

            previous = self.registry.get(fp.normalized_path)
            if previous is None or previous.status == "missing":
                plan.files_new.append(fp)
                continue

            if previous.signature == fp.signature:
                plan.files_unchanged.append(fp)
            else:
                plan.files_changed.append(fp)

        for known_path in self.registry.known_paths():
            if known_path not in seen_supported_paths:
                row = self.registry.get(known_path)
                if row and row.status != "missing":
                    plan.files_missing.append(known_path)

        return plan

    def ingest(self, force: bool = False) -> IncrementalIngestionReport:
        """
        Execute incremental ingestion.

        force=True reprocesses all supported files but still avoids wiping the
        whole database. Existing rows for selected files are removed first.
        """

        plan = self.plan()
        selected = plan.selected_for_ingestion(force=force)
        errors: list[str] = []
        removed_count = 0

        # Deleted/missing files: remove their current metadata immediately.
        for missing_path in plan.files_missing:
            removed_count += self.remove_file_records(missing_path)
            self.registry.mark_missing(missing_path)

        # Changed/force-selected files: remove old rows before reingesting.
        files_to_remove = selected if force else plan.files_changed
        for fp in files_to_remove:
            removed_count += self.remove_file_records(fp.normalized_path)

        documents_created = 0
        chunks_created = 0
        ingestion_report_dict: dict | None = None

        if selected:
            try:
                ingestion_report_dict = self._ingest_selected_files(selected)
                documents_created = int(ingestion_report_dict.get("documents_created", 0))
                chunks_created = int(ingestion_report_dict.get("chunks_created", 0))
                errors.extend(ingestion_report_dict.get("errors", []) or [])
            except Exception as exc:
                errors.append(f"Incremental ingestion failed: {exc}")

        # Update registry for unchanged files seen in this scan.
        for fp in plan.files_unchanged:
            if force:
                continue
            previous = self.registry.get(fp.normalized_path)
            self.registry.upsert_seen(
                fingerprint=fp,
                status="unchanged",
                document_id=previous.document_id if previous else None,
                chunks_count=previous.chunks_count if previous else 0,
                error=None,
            )

        # Update registry for selected files based on final DB state.
        for fp in selected:
            document_id, chunk_count = self.lookup_document_for_file(fp.normalized_path)
            if chunk_count > 0:
                self.registry.mark_ingested(fp, document_id=document_id, chunks_count=chunk_count)
            else:
                self.registry.mark_ingested(
                    fp,
                    document_id=document_id,
                    chunks_count=0,
                    error="No chunks found after ingestion.",
                )

        return IncrementalIngestionReport(
            mode="force" if force else "incremental",
            input_dir=str(self.input_dir),
            metadata_db=str(self.metadata_db_path),
            files_seen=plan.files_seen,
            files_new=len(plan.files_new),
            files_changed=len(plan.files_changed),
            files_unchanged=len(plan.files_unchanged),
            files_missing=len(plan.files_missing),
            files_selected_for_ingestion=len(selected),
            files_removed_from_index=removed_count,
            documents_created=documents_created,
            chunks_created=chunks_created,
            errors=errors,
            ingestion_report=ingestion_report_dict,
            index_status="stale" if selected or plan.files_missing else "fresh",
        )

    def _ingest_selected_files(self, selected: list[FileFingerprint]) -> dict:
        """
        Ingest selected files through a temporary mirrored input directory.
        """

        self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="rags_incremental_", dir=self.work_dir) as temp_root_str:
            temp_root = Path(temp_root_str)
            staged_to_original: dict[str, FileFingerprint] = {}

            for fp in selected:
                source = Path(fp.file_path)
                relative = Path(fp.normalized_path)
                target = temp_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                staged_to_original[normalize_relative_path(target, temp_root)] = fp

            report = run_ingestion(
                input_dir=str(temp_root),
                output_dir=str(self.output_dir),
                metadata_db_path=str(self.metadata_db_path),
                max_chars=self.max_chars,
                overlap_chars=self.overlap_chars,
                vision_config=self.vision_config,
                ocr_config=self.ocr_config,
                auto_metadata_config=self.auto_metadata_config,
                llm_client=self.llm_client,
            )

            report_dict = report.__dict__ if hasattr(report, "__dict__") else dict(report)
            self.rewrite_staged_paths(temp_root=temp_root, staged_to_original=staged_to_original)
            return json_safe(report_dict)

    def rewrite_staged_paths(
        self,
        temp_root: Path,
        staged_to_original: dict[str, FileFingerprint],
    ) -> None:
        """
        Replace temporary source_file paths with real input paths in SQLite.
        """

        if not self.metadata_db_path.exists():
            return

        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.row_factory = sqlite3.Row

            for table in ("documents", "chunks"):
                if not table_exists(conn, table):
                    continue
                if not column_exists(conn, table, "source_file"):
                    continue

                rows = conn.execute(
                    f"SELECT rowid, source_file FROM {table};"
                ).fetchall()

                for row in rows:
                    source_file = row["source_file"]
                    staged_key = normalize_stored_path(source_file, temp_root)
                    original = staged_to_original.get(staged_key)
                    if not original:
                        continue

                    conn.execute(
                        f"UPDATE {table} SET source_file = ? WHERE rowid = ?;",
                        (original.file_path, row["rowid"]),
                    )

            # Keep the FTS table consistent if it already exists. It is still
            # recommended to rebuild keyword index after ingestion.
            if table_exists(conn, "chunks_fts") and column_exists(conn, "chunks_fts", "source_file"):
                rows = conn.execute("SELECT rowid, source_file FROM chunks_fts;").fetchall()
                for row in rows:
                    staged_key = normalize_stored_path(row["source_file"], temp_root)
                    original = staged_to_original.get(staged_key)
                    if original:
                        conn.execute(
                            "UPDATE chunks_fts SET source_file = ? WHERE rowid = ?;",
                            (original.file_path, row["rowid"]),
                        )

            conn.commit()

    def remove_file_records(self, normalized_path: str) -> int:
        """
        Remove document/chunk/FTS rows for one normalized input path.

        Returns the number of chunk rows removed.
        """

        if not self.metadata_db_path.exists():
            return 0

        removed_chunks = 0

        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.row_factory = sqlite3.Row

            chunk_ids: list[str] = []
            document_ids: set[str] = set()

            if table_exists(conn, "chunks") and column_exists(conn, "chunks", "source_file"):
                rows = conn.execute(
                    "SELECT chunk_id, document_id, source_file FROM chunks;"
                ).fetchall()
                for row in rows:
                    if normalize_stored_path(row["source_file"], self.input_dir) == normalized_path:
                        chunk_ids.append(row["chunk_id"])
                        if row["document_id"]:
                            document_ids.add(row["document_id"])

            if table_exists(conn, "documents") and column_exists(conn, "documents", "source_file"):
                rows = conn.execute(
                    "SELECT document_id, source_file FROM documents;"
                ).fetchall()
                for row in rows:
                    if normalize_stored_path(row["source_file"], self.input_dir) == normalized_path:
                        document_ids.add(row["document_id"])

            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                removed_chunks = conn.execute(
                    f"DELETE FROM chunks WHERE chunk_id IN ({placeholders});",
                    chunk_ids,
                ).rowcount

                if table_exists(conn, "chunks_fts") and column_exists(conn, "chunks_fts", "chunk_id"):
                    conn.execute(
                        f"DELETE FROM chunks_fts WHERE chunk_id IN ({placeholders});",
                        chunk_ids,
                    )

            if document_ids:
                document_id_list = list(document_ids)
                placeholders = ",".join("?" for _ in document_id_list)
                conn.execute(
                    f"DELETE FROM documents WHERE document_id IN ({placeholders});",
                    document_id_list,
                )

            conn.commit()

        return max(removed_chunks, 0)

    def lookup_document_for_file(self, normalized_path: str) -> tuple[str | None, int]:
        """
        Find document id and chunk count currently associated with one file.
        """

        if not self.metadata_db_path.exists():
            return None, 0

        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.row_factory = sqlite3.Row

            if not table_exists(conn, "chunks") or not column_exists(conn, "chunks", "source_file"):
                return None, 0

            rows = conn.execute(
                "SELECT document_id, source_file FROM chunks;"
            ).fetchall()

        document_counts: dict[str, int] = {}
        for row in rows:
            if normalize_stored_path(row["source_file"], self.input_dir) == normalized_path:
                document_id = row["document_id"]
                document_counts[document_id] = document_counts.get(document_id, 0) + 1

        if not document_counts:
            return None, 0

        document_id, count = max(document_counts.items(), key=lambda item: item[1])
        return document_id, count


def is_runtime_or_hidden_file(path: Path) -> bool:
    """
    Skip internal/runtime files that should not be ingested.
    """

    if path.name.startswith("."):
        return True
    if "__pycache__" in path.parts:
        return True
    return False


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    except sqlite3.OperationalError:
        return False
    return any(row[1] == column_name for row in rows)


def json_safe(value):
    """
    Convert dataclasses/paths into JSON-safe values for CLI output.
    """

    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return str(value)
