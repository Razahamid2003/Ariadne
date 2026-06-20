"""Identifier and hashing helpers for ingestion.

Purpose
-------
Generates stable, deterministic identifiers for documents and chunks so the same
input always produces the same IDs.

What it does
------------
Normalizes text into safe identifiers, computes short deterministic hashes, and
builds document IDs from file paths and chunk IDs from a document ID plus chunk
number.
"""

import hashlib
import re
from pathlib import Path


def normalize_identifier(value: str) -> str:
    """
    Convert arbitrary text into a safe lowercase identifier.

    Example:
        "Centrifuge C-900 Error E-112.pdf"
        -> "centrifuge-c-900-error-e-112-pdf"
    """

    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "unknown"


def short_hash(text: str, length: int = 10) -> str:
    """
    Return a deterministic short hash for text.
    """

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def document_id_from_path(path: str | Path) -> str:
    """
    Create a stable document ID from a file path.

    The path hash prevents collisions when different source folders contain
    files with the same name.
    """

    file_path = Path(path)
    normalized_name = normalize_identifier(file_path.stem)
    path_hash = short_hash(str(file_path).replace("\\", "/"))
    return f"doc-{normalized_name}-{path_hash}"


def chunk_id(document_id: str, chunk_index: int) -> str:
    """
    Create a stable chunk ID from document ID and chunk number.
    """

    return f"{document_id}-chunk-{chunk_index:04d}"