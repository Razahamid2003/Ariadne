"""Inspect the metadata database.

Purpose
-------
Read-only check of documents, chunks, ingestion runs, and recent chunk previews.

Usage
-----
    python scripts/inspect_metadata.py --config config/client.yaml
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.persistence.sqlite_metadata_store import SQLiteMetadataStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect metadata database.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of recent chunks to preview.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    store = SQLiteMetadataStore(settings.paths.metadata_db)

    summary = {
        "metadata_db": settings.paths.metadata_db,
        "documents": store.count_documents(),
        "chunks": store.count_chunks(),
        "ingestion_runs": store.count_ingestion_runs(),
        "recent_chunks": store.list_recent_chunks(limit=args.limit),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())