"""Run ingestion.

Purpose
-------
Ingests all supported files from the input directory, writes inspection outputs,
and stores documents and chunks in the metadata database.

Usage
-----
    python scripts/ingest.py --config config/client.yaml
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.ingestion.pipeline import run_ingestion


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified ingestion.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    report = run_ingestion(
        input_dir=settings.paths.input_data,
        output_dir=settings.paths.processed_data,
        metadata_db_path=settings.paths.metadata_db,
        max_chars=settings.ingestion.max_chars,
        overlap_chars=settings.ingestion.overlap_chars,
        vision_config=settings.vision,
        ocr_config=settings.ocr,
    )

    print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))

    if report.errors:
        print("[WARN] Ingestion completed with warnings.")
    else:
        print("[PASS] Ingestion completed successfully.")

    print(f"[INFO] Metadata DB: {settings.paths.metadata_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())