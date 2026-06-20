"""Test OCR on one image.

Purpose
-------
Runs OCR on a single image and prints the extracted text, without ingesting or
writing anything.

Usage
-----
    python scripts/test_ocr.py --config config/client.yaml --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.ocr.local_ocr import LocalOcrClient


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_first_image(input_dir: str | Path) -> Path | None:
    root = Path(input_dir)

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local OCR.")
    parser.add_argument("--config", default="config/client.yaml")
    parser.add_argument("--image", default=None)

    args = parser.parse_args()
    settings = load_settings(args.config)

    image_path = Path(args.image) if args.image else find_first_image(settings.paths.input_data)

    if image_path is None:
        print("[FAIL] No image found.")
        return 1

    client = LocalOcrClient(settings.ocr)

    print(f"[INFO] OCR enabled: {settings.ocr.enabled}")
    print(f"[INFO] OCR provider: {settings.ocr.provider}")
    print(f"[INFO] Tesseract command: {settings.ocr.tesseract_cmd}")
    print(f"[INFO] Image: {image_path}")

    result = client.ocr_image_file(image_path)

    output = {
        "status": result.status,
        "image": str(image_path),
        "text": result.text,
        "error": result.error,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    if result.status != "ok":
        print("[WARN] OCR did not produce usable text.")
        return 1

    print("[PASS] OCR test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())