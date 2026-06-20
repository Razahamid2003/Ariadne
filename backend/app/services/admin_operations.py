"""Admin operations.

Purpose
-------
Implements the heavy maintenance actions the admin API and background jobs call:
ingestion, index building, and cleanup.

What it does
------------
Plans and runs incremental ingestion, builds the keyword and vector indexes, and
cleans runtime outputs, with retries for transient file locks on Windows.

Flow
----
Each operation loads settings, performs its step against the database and index
files, and returns a result. A rebuild chains clean, ingest, count, keyword index,
and vector index in order.
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.embeddings.local_embedding_model import SentenceTransformersEmbeddingModel
from backend.app.indexing.vector_index import LocalVectorIndex
from backend.app.ingestion.pipeline import run_ingestion
from backend.app.intake.archive_extractor import extract_archives
from backend.app.intake.incremental_ingestion import IncrementalIngestionManager
from backend.app.persistence.sqlite_metadata_store import SQLiteMetadataStore
from backend.app.retrieval.keyword_index import KeywordSearchIndex


def plan_ingest(settings: Settings) -> dict[str, Any]:
    manager = _incremental_manager(settings)
    return manager.plan().to_dict()


def run_incremental_ingest(settings: Settings, force: bool = False) -> dict[str, Any]:
    manager = _incremental_manager(settings)
    report = manager.ingest(force=force)
    return report.to_dict()


def build_keyword_index(settings: Settings) -> dict[str, Any]:
    report = KeywordSearchIndex(
        metadata_db_path=settings.paths.metadata_db,
        table_name=settings.retrieval.keyword_table,
    ).rebuild()
    return report.to_dict()


def build_vector_index(settings: Settings) -> dict[str, Any]:
    store = SQLiteMetadataStore(settings.paths.metadata_db)
    chunks = store.list_chunks_for_indexing()

    if not chunks:
        raise RuntimeError("No chunks found. Run ingestion first.")

    texts = [chunk["text"] for chunk in chunks]
    embedding_model = SentenceTransformersEmbeddingModel(settings.embeddings)
    batch = embedding_model.encode(texts)

    metadata_rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_system": chunk["source_system"],
            "source_file": chunk["source_file"],
            "record_type": chunk["record_type"],
            "title": chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "sensitivity": chunk["sensitivity"],
            "citation_label": chunk["citation_label"],
            "text": chunk["text"],
            "metadata_json": chunk["metadata_json"],
        }
        for chunk in chunks
    ]

    index = LocalVectorIndex(
        index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
    )
    index.save(batch.vectors, metadata_rows)

    return {
        "chunks_indexed": batch.count,
        "embedding_dimension": batch.dimension,
        "index_dir": settings.vector_index.index_dir,
        "embeddings_path": str(index.embeddings_path),
        "metadata_path": str(index.metadata_path),
    }


def clean_runtime_outputs(settings: Settings, processed: bool, metadata: bool, vector: bool) -> dict[str, Any]:
    processed_dir = Path(settings.paths.processed_data)
    metadata_db = Path(settings.paths.metadata_db)
    vector_dir = Path(settings.vector_index.index_dir)
    removed: list[str] = []

    if processed and processed_dir.exists():
        for pattern in ("*.json", "*.jsonl"):
            for path in processed_dir.glob(pattern):
                _unlink_with_retries(path)
                removed.append(str(path))

    if metadata:
        for path in _sqlite_file_family(metadata_db):
            if path.exists():
                _unlink_with_retries(path)
                removed.append(str(path))

    if vector and vector_dir.exists():
        _rmtree_with_retries(vector_dir)
        removed.append(str(vector_dir))

    return {
        "processed_cleaned": processed,
        "metadata_cleaned": metadata,
        "vector_cleaned": vector,
        "removed": removed,
    }


def _sqlite_file_family(db_path: Path) -> list[Path]:
    """Return SQLite sidecar files that must be removed with the DB."""

    return [
        db_path,
        Path(f"{db_path}-wal"),
        Path(f"{db_path}-shm"),
        Path(f"{db_path}-journal"),
    ]


def _unlink_with_retries(path: Path, attempts: int = 12, delay_seconds: float = 0.35) -> None:
    """Delete a file, allowing brief Windows file-locks to clear."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            path.unlink(missing_ok=True)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc

        gc.collect()
        if attempt < attempts:
            time.sleep(delay_seconds)

    raise RuntimeError(
        f"Could not replace {path}. Ariadne is still holding the file or another "
        "program has it open. Close active source viewers/SQLite tools and retry the rebuild."
    ) from last_error


def _rmtree_with_retries(path: Path, attempts: int = 12, delay_seconds: float = 0.35) -> None:
    """Remove a directory tree, allowing brief Windows file-locks to clear."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc

        gc.collect()
        if attempt < attempts:
            time.sleep(delay_seconds)

    raise RuntimeError(
        f"Could not replace {path}. Ariadne is still holding an index file or another "
        "program has it open. Close active file viewers and retry the rebuild."
    ) from last_error


def run_rebuild(
    settings: Settings,
    fresh: bool = False,
    extract: bool = False,
    clear_extract: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    if fresh:
        steps.append({"step": "clean", "result": clean_runtime_outputs(settings, True, True, True)})

    if extract:
        archive_report = extract_archives(
            input_dir=settings.paths.input_data,
            extract_dir=settings.archives.extract_dir,
            clear_extract_dir=clear_extract,
        )
        steps.append({"step": "extract", "result": archive_report.__dict__})

    ingest_result = run_incremental_ingest(settings, force=False)
    steps.append({"step": "ingest", "result": ingest_result})

    keyword_result = build_keyword_index(settings)
    steps.append({"step": "build_keyword_index", "result": keyword_result})

    vector_result = build_vector_index(settings)
    steps.append({"step": "build_vector_index", "result": vector_result})

    return {
        "status": "completed",
        "fresh": fresh,
        "extract": extract,
        "clear_extract": clear_extract,
        "steps": steps,
    }


def _build_ingestion_llm_client(settings: Settings):
    """Build an LLM client for auto-metadata classification, or None if the
    feature is disabled. Kept lazy so ingestion without auto-metadata needs no LLM."""
    auto = getattr(settings.ingestion, "auto_metadata", None)
    if not auto or not getattr(auto, "enabled", False):
        return None
    try:
        from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient
        return OpenAICompatibleLLMClient(settings.llm)
    except Exception:
        return None


def _incremental_manager(settings: Settings) -> IncrementalIngestionManager:
    return IncrementalIngestionManager(
        input_dir=settings.paths.input_data,
        output_dir=settings.paths.processed_data,
        metadata_db_path=settings.paths.metadata_db,
        max_chars=settings.ingestion.max_chars,
        overlap_chars=settings.ingestion.overlap_chars,
        vision_config=settings.vision,
        ocr_config=settings.ocr,
        work_dir=settings.file_tracking.work_dir,
        auto_metadata_config=getattr(settings.ingestion, "auto_metadata", None),
        llm_client=_build_ingestion_llm_client(settings),
    )
