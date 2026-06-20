"""Caption budget.

Purpose
-------
Caps how many image captions a single ingestion run may request, so a large batch
of images cannot make ingestion run for an unreasonable time.

What it does
------------
Tracks separate limits for image captions and PDF-page captions and reports whether
each is still within budget.
"""

from dataclasses import dataclass

from backend.app.core.config import VisionConfig


@dataclass
class CaptionBudget:
    """
    Tracks how many vision captions have been used in one ingestion run.
    """

    max_images: int
    max_pdf_pages: int
    images_used: int = 0
    pdf_pages_used: int = 0

    @classmethod
    def from_config(cls, config: VisionConfig) -> "CaptionBudget":
        return cls(
            max_images=config.max_images_per_run,
            max_pdf_pages=config.max_pdf_pages_per_run,
        )

    def can_caption_image(self) -> bool:
        """
        Return True if another standalone image can be captioned.
        """

        return self.images_used < self.max_images

    def can_caption_pdf_page(self) -> bool:
        """
        Return True if another image-only PDF page can be captioned.
        """

        return self.pdf_pages_used < self.max_pdf_pages

    def use_image_caption(self) -> None:
        """
        Count one standalone image caption.
        """

        self.images_used += 1

    def use_pdf_page_caption(self) -> None:
        """
        Count one image-only PDF page caption.
        """

        self.pdf_pages_used += 1