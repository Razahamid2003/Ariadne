"""OCR budget.

Purpose
-------
Caps how much OCR a single ingestion run may do, so large batches of images or
pages cannot accidentally make ingestion run for hours.

What it does
------------
Tracks separate limits for image OCR and PDF-page OCR and reports whether each is
still within budget before each call.
"""

from dataclasses import dataclass

from backend.app.core.config import OcrConfig


@dataclass
class OcrBudget:
    """
    Tracks OCR usage in one ingestion run.
    """

    max_images: int
    max_pdf_pages: int
    images_used: int = 0
    pdf_pages_used: int = 0

    @classmethod
    def from_config(cls, config: OcrConfig) -> "OcrBudget":
        return cls(
            max_images=config.max_images_per_run,
            max_pdf_pages=config.max_pdf_pages_per_run,
        )

    def can_ocr_image(self) -> bool:
        return self.images_used < self.max_images

    def can_ocr_pdf_page(self) -> bool:
        return self.pdf_pages_used < self.max_pdf_pages

    def use_image_ocr(self) -> None:
        self.images_used += 1

    def use_pdf_page_ocr(self) -> None:
        self.pdf_pages_used += 1