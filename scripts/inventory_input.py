"""Inventory input files.

Purpose
-------
Audits the input directory before ingestion, categorizing files by type so you know
what the corpus contains.

Usage
-----
    python scripts/inventory_input.py --config config/client.yaml
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.ingestion.registry import LoaderRegistry


ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"}
STRUCTURED_EXTENSIONS = {".csv", ".xlsx"}


def categorize_extension(extension: str) -> str:
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in DOCUMENT_EXTENSIONS:
        return "document"

    if extension in STRUCTURED_EXTENSIONS:
        return "structured"

    if extension == "":
        return "no_extension"

    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory input files.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Number of sample paths to show per extension.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    input_root = Path(settings.paths.input_data)
    registry = LoaderRegistry()
    supported_extensions = registry.supported_extensions()

    files = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and "__pycache__" not in path.parts
    ]

    extension_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    supported_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    samples_by_extension: dict[str, list[str]] = defaultdict(list)

    for path in files:
        extension = path.suffix.lower()
        category = categorize_extension(extension)

        extension_counts[extension or "[no extension]"] += 1
        category_counts[category] += 1

        if extension in supported_extensions:
            supported_counts[extension] += 1
        else:
            unsupported_counts[extension or "[no extension]"] += 1

        key = extension or "[no extension]"
        if len(samples_by_extension[key]) < args.sample_limit:
            samples_by_extension[key].append(str(path))

    summary = {
        "input_dir": str(input_root),
        "total_files": len(files),
        "supported_extensions": sorted(supported_extensions),
        "counts_by_category": dict(category_counts),
        "counts_by_extension": dict(sorted(extension_counts.items())),
        "supported_counts": dict(sorted(supported_counts.items())),
        "unsupported_counts": dict(sorted(unsupported_counts.items())),
        "samples_by_extension": dict(samples_by_extension),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if unsupported_counts:
        print("[WARN] Unsupported file types are present. Review unsupported_counts.")
    else:
        print("[PASS] All discovered file extensions are supported.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())