"""System status builder.

Purpose
-------
Assembles the runtime and index status shown in the readiness panel and returned by
the status endpoint.

What it does
------------
Reads document and chunk counts from the database, checks whether the keyword and
vector indexes exist, and reports the active model and deployment mode.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import Settings
from backend.app.indexing.index_status import get_index_lifecycle_status
from backend.app.intake.file_registry import FileRegistry
from backend.app.retrieval.keyword_index import KeywordSearchIndex


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def safe_count_table(conn: sqlite3.Connection, table_name: str) -> int:
    if not table_exists(conn, table_name):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0])


def build_system_status(settings: Settings, config_path: str = "config/client.yaml") -> dict[str, Any]:
    """Return browser/API-friendly runtime/index status."""

    metadata_db = Path(settings.paths.metadata_db)
    vector_dir = Path(settings.vector_index.index_dir)
    embeddings_path = vector_dir / settings.vector_index.embeddings_file
    metadata_path = vector_dir / settings.vector_index.metadata_file

    db_status: dict[str, Any] = {
        "exists": metadata_db.exists(),
        "path": str(metadata_db),
    }

    if metadata_db.exists():
        with sqlite3.connect(metadata_db) as conn:
            db_status["documents"] = safe_count_table(conn, "documents")
            db_status["chunks"] = safe_count_table(conn, "chunks")
            db_status["ingestion_runs"] = safe_count_table(conn, "ingestion_runs")

            if table_exists(conn, "chunks"):
                rows = conn.execute(
                    """
                    SELECT record_type, COUNT(*) AS count
                    FROM chunks
                    GROUP BY record_type
                    ORDER BY count DESC;
                    """
                ).fetchall()
            else:
                rows = []

            db_status["record_types"] = [
                {"record_type": row[0], "count": row[1]}
                for row in rows
            ]

    vector_status: dict[str, Any] = {
        "index_dir": str(vector_dir),
        "embeddings_exists": embeddings_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }

    if embeddings_path.exists():
        vectors = np.load(embeddings_path)
        vector_status["vector_shape"] = list(vectors.shape)

    keyword_status = KeywordSearchIndex(
        metadata_db_path=settings.paths.metadata_db,
        table_name=settings.retrieval.keyword_table,
    ).table_status()

    registry = FileRegistry(settings.paths.metadata_db)
    file_registry_status = registry.summary() if metadata_db.exists() else {
        "exists": False,
        "total_files_tracked": 0,
        "status_counts": {},
    }

    lifecycle_status = get_index_lifecycle_status(
        metadata_db_path=settings.paths.metadata_db,
        vector_index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
        keyword_table=settings.retrieval.keyword_table,
    )

    return {
        "config": config_path,
        "deployment": {
            "mode": settings.deployment.mode,
            "host": settings.app.host,
            "port": settings.app.port,
            "offline_mode": settings.app.offline_mode,
        },
        "llm": {
            "provider": settings.llm.provider,
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
        },
        "ocr": {
            "enabled": settings.ocr.enabled,
            "provider": settings.ocr.provider,
            "ocr_images": settings.ocr.ocr_images,
            "ocr_pdf_pages": settings.ocr.ocr_pdf_pages,
            "max_images_per_run": settings.ocr.max_images_per_run,
            "max_pdf_pages_per_run": settings.ocr.max_pdf_pages_per_run,
        },
        "vision": {
            "enabled": settings.vision.enabled,
            "provider": settings.vision.provider,
            "base_url": settings.vision.base_url,
            "model": settings.vision.model,
            "caption_images": settings.vision.caption_images,
            "caption_pdf_pages": settings.vision.caption_pdf_pages,
            "max_images_per_run": settings.vision.max_images_per_run,
            "max_pdf_pages_per_run": settings.vision.max_pdf_pages_per_run,
        },
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model_name_or_path,
            "device": settings.embeddings.device,
            "batch_size": settings.embeddings.batch_size,
        },
        "retrieval": {
            "keyword_table": settings.retrieval.keyword_table,
            "vector_top_k": settings.retrieval.vector_top_k,
            "keyword_top_k": settings.retrieval.keyword_top_k,
            "final_top_k": settings.retrieval.final_top_k,
            "vector_weight": settings.retrieval.vector_weight,
            "keyword_weight": settings.retrieval.keyword_weight,
            "exact_match_boost": settings.retrieval.exact_match_boost,
            "deduplicate": settings.retrieval.deduplicate,
        },
        "rag": {
            "max_context_chunks": settings.rag.max_context_chunks,
            "max_context_chars": settings.rag.max_context_chars,
            "max_chars_per_chunk": settings.rag.max_chars_per_chunk,
            "drop_vector_only_below_score": settings.rag.drop_vector_only_below_score,
            "min_retrieval_confidence": settings.rag.min_retrieval_confidence,
            "require_citations": settings.rag.require_citations,
            "retry_on_invalid_citations": settings.rag.retry_on_invalid_citations,
            "answer_temperature": settings.rag.answer_temperature,
            "answer_max_tokens": settings.rag.answer_max_tokens,
        },
        "runtime": {
            "max_chat_concurrency": settings.runtime.max_chat_concurrency,
            "max_search_concurrency": settings.runtime.max_search_concurrency,
            "max_admin_jobs": settings.runtime.max_admin_jobs,
            "reject_chat_during_rebuild": settings.runtime.reject_chat_during_rebuild,
        },
        "file_tracking": {
            "enabled": settings.file_tracking.enabled,
            "work_dir": settings.file_tracking.work_dir,
            "registry": file_registry_status,
        },
        "metadata_db": db_status,
        "keyword_index": keyword_status,
        "vector_index": vector_status,
        "index_lifecycle": lifecycle_status,
    }
