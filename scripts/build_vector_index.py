"""Build the vector index.

Purpose
-------
Reads chunks from the metadata database, generates embeddings, and writes the local
vector index.

Usage
-----
    python scripts/build_vector_index.py --config config/client.yaml
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.embeddings.local_embedding_model import SentenceTransformersEmbeddingModel
from backend.app.indexing.vector_index import LocalVectorIndex
from backend.app.persistence.sqlite_metadata_store import SQLiteMetadataStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local vector index.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)

    store = SQLiteMetadataStore(settings.paths.metadata_db)
    chunks = store.list_chunks_for_indexing()

    if not chunks:
        print("[FAIL] No chunks found. Run ingestion first.")
        return 1

    texts = [chunk["text"] for chunk in chunks]

    print(f"[INFO] Chunks to embed: {len(texts)}")
    print(f"[INFO] Embedding provider: {settings.embeddings.provider}")
    print(f"[INFO] Embedding model: {settings.embeddings.model_name_or_path}")
    print(f"[INFO] Device: {settings.embeddings.device}")
    print(f"[INFO] Batch size: {settings.embeddings.batch_size}")

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

    summary = {
        "chunks_indexed": batch.count,
        "embedding_dimension": batch.dimension,
        "index_dir": settings.vector_index.index_dir,
        "embeddings_path": str(index.embeddings_path),
        "metadata_path": str(index.metadata_path),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[PASS] Vector index built successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())