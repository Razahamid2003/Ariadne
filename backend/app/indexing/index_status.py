"""Index lifecycle status helpers.

Purpose
-------
Reports, in one place, whether the metadata database, keyword index, and vector
index exist and how many rows/vectors they contain.

Flow
----
Inspects the SQLite database and the vector index files on disk and returns a
compact status used by the readiness panel and CLI status command.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


def get_index_lifecycle_status(
    metadata_db_path: str | Path,
    vector_index_dir: str | Path,
    embeddings_file: str,
    metadata_file: str,
    keyword_table: str = "chunks_fts",
) -> dict[str, Any]:
    """
    Return compact status for metadata, keyword index, and vector index.
    """

    metadata_db = Path(metadata_db_path)
    vector_dir = Path(vector_index_dir)
    embeddings_path = vector_dir / embeddings_file
    vector_metadata_path = vector_dir / metadata_file

    chunks_count = 0
    documents_count = 0
    keyword_rows = 0
    keyword_exists = False

    if metadata_db.exists():
        with sqlite3.connect(metadata_db) as conn:
            documents_count = count_table_rows(conn, "documents")
            chunks_count = count_table_rows(conn, "chunks")
            keyword_exists = table_exists(conn, keyword_table)
            keyword_rows = count_table_rows(conn, keyword_table) if keyword_exists else 0

    vector_rows = 0
    vector_dimension = None
    vector_metadata_rows = 0

    if embeddings_path.exists():
        vectors = np.load(embeddings_path)
        if len(vectors.shape) == 2:
            vector_rows = int(vectors.shape[0])
            vector_dimension = int(vectors.shape[1])

    if vector_metadata_path.exists():
        with vector_metadata_path.open("r", encoding="utf-8") as file:
            vector_metadata_rows = sum(1 for line in file if line.strip())

    keyword_status = "fresh" if keyword_exists and keyword_rows == chunks_count else "stale"
    vector_status = (
        "fresh"
        if vector_rows == chunks_count and vector_metadata_rows == chunks_count and chunks_count > 0
        else "stale"
    )

    return {
        "metadata": {
            "exists": metadata_db.exists(),
            "path": str(metadata_db),
            "documents": documents_count,
            "chunks": chunks_count,
        },
        "keyword_index": {
            "table": keyword_table,
            "exists": keyword_exists,
            "rows": keyword_rows,
            "status": keyword_status,
        },
        "vector_index": {
            "index_dir": str(vector_dir),
            "embeddings_exists": embeddings_path.exists(),
            "metadata_exists": vector_metadata_path.exists(),
            "vector_rows": vector_rows,
            "vector_dimension": vector_dimension,
            "metadata_rows": vector_metadata_rows,
            "status": vector_status,
        },
        "overall_status": "fresh" if keyword_status == "fresh" and vector_status == "fresh" else "stale",
    }


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def count_table_rows(conn: sqlite3.Connection, table_name: str) -> int:
    if not table_exists(conn, table_name):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0])
