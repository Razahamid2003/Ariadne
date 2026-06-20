"""Count chunk record types.

Purpose
-------
Prints how many chunks of each record type are stored, a quick way to sanity-check
what was ingested.

Usage
-----
    python scripts/count_record_types.py --config config/client.yaml
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Count chunk record types.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)

    with sqlite3.connect(settings.paths.metadata_db) as conn:
        rows = conn.execute(
            """
            SELECT record_type, COUNT(*) AS count
            FROM chunks
            GROUP BY record_type
            ORDER BY count DESC;
            """
        ).fetchall()

    output = [{"record_type": row[0], "count": row[1]} for row in rows]
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())