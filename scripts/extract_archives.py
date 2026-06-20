"""Extract archives before ingestion.

Purpose
-------
Expands supported archives into the configured extraction folder so ingestion can
process their contents.

Usage
-----
    python scripts/extract_archives.py --config config/client.yaml
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.intake.archive_extractor import extract_archives


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract archives before ingestion.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear extraction folder before extracting archives.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    if not settings.archives.enabled:
        print("[INFO] Archive extraction is disabled in config.")
        return 0

    report = extract_archives(
        input_dir=settings.paths.input_data,
        extract_dir=settings.archives.extract_dir,
        clear_extract_dir=args.clear,
    )

    print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))

    if report.errors:
        print("[WARN] Archive extraction completed with warnings.")
    else:
        print("[PASS] Archive extraction completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())