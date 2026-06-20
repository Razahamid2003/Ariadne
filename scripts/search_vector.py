"""Search the vector index.

Purpose
-------
Runs a quick semantic search against the vector index to test retrieval directly.

Usage
-----
    python scripts/search_vector.py --config config/client.yaml --query "your question"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Search local vector index.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--source-system",
        default=None,
        help="Optional source system filter, such as DOCS, IMAGES, MEALS, HEADS, GEAR.",
    )
    parser.add_argument(
        "--record-type",
        default=None,
        help="Optional record type filter, such as pdf_page, image_caption, csv_row.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    print(f"[INFO] Query: {args.query}")
    print(f"[INFO] Top K: {args.top_k}")
    print(f"[INFO] Source filter: {args.source_system}")
    print(f"[INFO] Record type filter: {args.record_type}")
    print(f"[INFO] Embedding model: {settings.embeddings.model_name_or_path}")

    embedding_model = SentenceTransformersEmbeddingModel(settings.embeddings)
    query_batch = embedding_model.encode([args.query], show_progress_bar=False)

    index = LocalVectorIndex(
        index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
    )

    results = index.search(
        query_vector=query_batch.vectors,
        top_k=args.top_k,
        source_system=args.source_system,
        record_type=args.record_type,
    )

    output = [
        {
            "rank": rank,
            "score": round(result.score, 4),
            "chunk_id": result.chunk_id,
            "source_system": result.source_system,
            "record_type": result.record_type,
            "title": result.title,
            "source_file": result.source_file,
            "citation_label": result.citation_label,
            "text_preview": result.text[:500],
        }
        for rank, result in enumerate(results, start=1)
    ]

    print(json.dumps(output, indent=2, ensure_ascii=False))

    if not output:
        print("[WARN] No vector results found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())