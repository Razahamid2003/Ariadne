"""Test the local vision model.

Purpose
-------
Sends one image to the configured local vision model and prints the caption.

Usage
-----
    python scripts/test_vision.py --config config/client.yaml --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.vision.local_vision_client import LocalVisionClient


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
    """
    Find the first supported image under input_dir.
    """

    root = Path(input_dir)

    if not root.exists():
        return None

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Test local vision model.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image path. If omitted, the first image in input_data is used.",
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    if not settings.vision.enabled or settings.vision.mode != "caption":
        print("[FAIL] Vision captioning is not enabled in config.")
        print("Set vision.enabled=true and vision.mode=caption.")
        return 1

    image_path = Path(args.image) if args.image else find_first_image(settings.paths.input_data)

    if image_path is None:
        print("[FAIL] No supported image found.")
        return 1

    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        return 1

    client = LocalVisionClient(settings.vision)

    print(f"[INFO] Vision provider: {settings.vision.provider}")
    print(f"[INFO] Vision model: {settings.vision.model}")
    print(f"[INFO] Image: {image_path}")

    result = client.caption_image_file(
        image_path=image_path,
        prompt=(
            "Describe this image for a retrieval system. "
            "Focus on visible objects, equipment, text if readable, logos if visible, "
            "and any useful identifying details. Do not invent details."
        ),
    )

    output = {
        "status": result.status,
        "image": str(image_path),
        "model": settings.vision.model,
        "caption": result.caption,
        "error": result.error,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

    if result.status != "ok":
        print("[FAIL] Vision model test failed.")
        return 1

    print("[PASS] Vision model test succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())