"""Local OCR client.

Purpose
-------
Extracts visible text from images using a local OCR engine, so scanned pages,
product cards, labels, and screenshots become searchable.

What it does
------------
Wraps the local OCR engine, prepares images for better recognition, runs OCR on
files or raw bytes, cleans the result, and judges whether the extracted text is
usable.

Flow
----
An image is pre-processed, passed to the engine, and the returned text is cleaned
and checked; unusable output is discarded so noise does not enter the index.
"""

import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from backend.app.core.config import OcrConfig


@dataclass(frozen=True)
class OcrResult:
    """
    OCR output.
    """

    status: str
    text: str
    error: str | None = None


class LocalOcrClient:
    """
    Local Tesseract OCR wrapper.
    """

    def __init__(self, config: OcrConfig):
        self.config = config
        self._configure_tesseract()

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.provider == "tesseract"

    def ocr_image_file(self, image_path: str | Path) -> OcrResult:
        """
        OCR an image file.
        """

        path = Path(image_path)

        if not path.exists():
            return OcrResult(
                status="error",
                text="",
                error=f"Image not found: {path}",
            )

        try:
            image_bytes = path.read_bytes()
            return self.ocr_image_bytes(image_bytes=image_bytes, image_label=path.name)

        except Exception as exc:
            return OcrResult(status="error", text="", error=str(exc))

    def ocr_image_bytes(self, image_bytes: bytes, image_label: str = "image") -> OcrResult:
        """
        OCR image bytes.
        """

        if not self.is_enabled():
            return OcrResult(
                status="disabled",
                text="",
                error="OCR is disabled.",
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = self._prepare_image_for_ocr(image)

            tesseract_config = f"--psm {self.config.psm}"

            raw_text = pytesseract.image_to_string(
                image,
                lang=self.config.languages,
                config=tesseract_config,
            )

            cleaned_text = self._clean_text(raw_text)

            if not self.is_usable_text(cleaned_text):
                return OcrResult(
                    status="weak",
                    text=cleaned_text,
                    error=f"OCR text failed quality checks for {image_label}.",
                )

            return OcrResult(status="ok", text=cleaned_text, error=None)

        except Exception as exc:
            return OcrResult(status="error", text="", error=str(exc))

    def is_usable_text(self, text: str) -> bool:
        """
        Decide whether OCR text is useful enough to index.

        This rejects common garbage OCR from object photos, blurry images,
        low-text exhibition photos, and random visual noise.
        """

        cleaned = text.strip()

        if len(cleaned) < self.config.min_text_chars:
            return False

        alnum_chars = [char for char in cleaned if char.isalnum()]
        alpha_chars = [char for char in cleaned if char.isalpha()]
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]{2,}", cleaned)

        if len(alnum_chars) < self.config.min_text_chars:
            return False

        if len(alpha_chars) < 10:
            return False

        if len(words) < 3:
            return False

        visible_chars = [char for char in cleaned if not char.isspace()]
        if not visible_chars:
            return False

        alnum_ratio = len(alnum_chars) / len(visible_chars)

        # Garbage OCR often has too much punctuation, quotes, random symbols.
        if alnum_ratio < 0.55:
            return False

        average_word_length = sum(len(word) for word in words) / max(len(words), 1)

        if average_word_length < 3:
            return False

        return True

    def _configure_tesseract(self) -> None:
        """
        Configure Tesseract executable path.

        Priority:
            1. config.tesseract_cmd
            2. PATH
            3. common Windows install path
        """

        if self.config.tesseract_cmd:
            candidate = Path(self.config.tesseract_cmd)

            if candidate.exists():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return

        path_candidate = shutil.which("tesseract")

        if path_candidate:
            pytesseract.pytesseract.tesseract_cmd = path_candidate
            return

        windows_candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

        if windows_candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(windows_candidate)

    @staticmethod
    def _prepare_image_for_ocr(image: Image.Image) -> Image.Image:
        """
        Prepare image for OCR.

        Conservative preprocessing:
            - convert to RGB
            - grayscale
            - auto contrast
            - upscale small images
        """

        image = image.convert("RGB")
        image = ImageOps.grayscale(image)
        image = ImageOps.autocontrast(image)

        width, height = image.size

        if width < 1200:
            scale = 1200 / max(width, 1)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size)

        return image

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize OCR text.
        """

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        cleaned = "\n".join(lines)

        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        return cleaned.strip()