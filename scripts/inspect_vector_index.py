"""Inspect the vector index.

Purpose
-------
Reports the shape and contents of the local vector index files.

Usage
-----
    python scripts/inspect_vector_index.py --config config/client.yaml
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local vector index.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)

    index_dir = Path(settings.vector_index.index_dir)
    embeddings_path = index_dir / settings.vector_index.embeddings_file
    metadata_path = index_dir / settings.vector_index.metadata_file

    if not embeddings_path.exists():
        print(f"[FAIL] Missing embeddings file: {embeddings_path}")
        return 1

    if not metadata_path.exists():
        print(f"[FAIL] Missing metadata file: {metadata_path}")
        return 1

    vectors = np.load(embeddings_path)

    metadata_rows = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                metadata_rows.append(json.loads(line))

    source_counts = Counter(row.get("source_system") for row in metadata_rows)
    record_type_counts = Counter(row.get("record_type") for row in metadata_rows)

    summary = {
        "index_dir": str(index_dir),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "vector_shape": list(vectors.shape),
        "metadata_rows": len(metadata_rows),
        "source_counts": dict(source_counts),
        "record_type_counts": dict(record_type_counts),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if vectors.shape[0] != len(metadata_rows):
        print("[FAIL] Vector count does not match metadata rows.")
        return 1

    print("[PASS] Vector index looks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())