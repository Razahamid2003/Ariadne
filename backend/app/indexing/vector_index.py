"""Local vector index.

Purpose
-------
A lightweight, file-backed vector index that stores chunk embeddings and finds the
most similar chunks for a query, with no external vector database.

What it does
------------
``LocalVectorIndex`` saves and loads embedding arrays plus their metadata and runs
cosine-similarity search, returning ranked results.

Flow
----
At build time, embeddings and metadata are written to the vector directory. At
query time the index is loaded, the query vector is compared against all stored
vectors, and the top matches are returned.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    """
    One vector search result.
    """

    score: float
    chunk_id: str
    document_id: str
    source_system: str
    source_file: str
    record_type: str
    title: str | None
    citation_label: str
    text: str
    metadata_json: str | None = None


class LocalVectorIndex:
    """
    Simple NumPy-backed vector index.
    """

    def __init__(
        self,
        index_dir: str | Path,
        embeddings_file: str = "embeddings.npy",
        metadata_file: str = "metadata.jsonl",
    ):
        self.index_dir = Path(index_dir)
        self.embeddings_path = self.index_dir / embeddings_file
        self.metadata_path = self.index_dir / metadata_file

    def save(self, vectors: np.ndarray, metadata_rows: list[dict[str, Any]]) -> None:
        """
        Save vectors and metadata to disk.

        Args:
            vectors:
                Shape: [num_chunks, embedding_dim]

            metadata_rows:
                One metadata row per vector.
        """

        if vectors.shape[0] != len(metadata_rows):
            raise ValueError(
                f"Vector count {vectors.shape[0]} does not match metadata count {len(metadata_rows)}"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.embeddings_path, vectors)

        with self.metadata_path.open("w", encoding="utf-8") as file:
            for row in metadata_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Load vectors and metadata from disk.
        """

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        vectors = np.load(self.embeddings_path)

        metadata_rows: list[dict[str, Any]] = []

        with self.metadata_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    metadata_rows.append(json.loads(line))

        if vectors.shape[0] != len(metadata_rows):
            raise ValueError(
                f"Vector count {vectors.shape[0]} does not match metadata count {len(metadata_rows)}"
            )

        return vectors, metadata_rows

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_system: str | None = None,
        record_type: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search the local vector index.

        Args:
            query_vector:
                Normalized query vector of shape [embedding_dim] or [1, embedding_dim].

            top_k:
                Number of results to return.

            source_system:
                Optional source-system filter.

            record_type:
                Optional record-type filter.
        """

        vectors, metadata_rows = self.load()

        if vectors.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32)

        if query.ndim == 2:
            query = query[0]

        if query.ndim != 1:
            raise ValueError(f"Expected query vector with 1 dimension, got shape {query.shape}")

        scores = vectors @ query

        candidate_indices = list(range(len(metadata_rows)))

        if source_system:
            wanted = source_system.upper()
            candidate_indices = [
                index
                for index in candidate_indices
                if str(metadata_rows[index].get("source_system", "")).upper() == wanted
            ]

        if record_type:
            candidate_indices = [
                index
                for index in candidate_indices
                if str(metadata_rows[index].get("record_type", "")) == record_type
            ]

        if not candidate_indices:
            return []

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        results: list[VectorSearchResult] = []

        for index in ranked_indices:
            row = metadata_rows[index]

            results.append(
                VectorSearchResult(
                    score=float(scores[index]),
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    source_system=row["source_system"],
                    source_file=row["source_file"],
                    record_type=row["record_type"],
                    title=row.get("title"),
                    citation_label=row["citation_label"],
                    text=row["text"],
                    metadata_json=row.get("metadata_json"),
                )
            )

        return results