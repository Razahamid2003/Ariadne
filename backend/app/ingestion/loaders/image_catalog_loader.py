"""Image loader with optional OCR and captioning.

Purpose
-------
Reads standalone image files and turns them into searchable records using their
visible text and, optionally, a generated caption.

Flow
----
For each image it attempts OCR to capture visible text, optionally requests a
caption from the local vision model, and builds a catalog record so the image is
represented in search even when no text is found.
"""

from pathlib import Path

from backend.app.core.config import OcrConfig, VisionConfig
from backend.app.ingestion.loaders.base import LoadedFile, LoadedRecord
from backend.app.ocr.local_ocr import LocalOcrClient
from backend.app.ocr.ocr_budget import OcrBudget
from backend.app.vision.caption_budget import CaptionBudget
from backend.app.vision.local_vision_client import LocalVisionClient


class ImageCatalogLoader:
    """
    Image loader with optional OCR and optional local vision captioning.
    """

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }

    def __init__(
        self,
        vision_config: VisionConfig | None = None,
        caption_budget: CaptionBudget | None = None,
        ocr_config: OcrConfig | None = None,
        ocr_budget: OcrBudget | None = None,
    ):
        self.vision_config = vision_config or VisionConfig()
        self.caption_budget = caption_budget or CaptionBudget.from_config(self.vision_config)
        self.vision_client = LocalVisionClient(self.vision_config)

        self.ocr_config = ocr_config or OcrConfig()
        self.ocr_budget = ocr_budget or OcrBudget.from_config(self.ocr_config)
        self.ocr_client = LocalOcrClient(self.ocr_config)

    def supports(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.supported_extensions

    def load(self, path: str | Path) -> LoadedFile:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(f"Unsupported image file type: {file_path.suffix}")

        records: list[LoadedRecord] = []

        ocr_record = self._try_ocr(file_path)
        if ocr_record:
            records.append(ocr_record)

        caption_record = self._try_caption(file_path)
        if caption_record:
            records.append(caption_record)

        if not records:
            records.append(self._build_catalog_record(file_path))

        metadata = self._base_metadata(file_path)
        metadata["records_created"] = len(records)

        return LoadedFile(
            source_path=file_path,
            records=records,
            document_metadata=metadata,
        )

    def _try_ocr(self, file_path: Path) -> LoadedRecord | None:
        """
        Try OCR before vision captioning.
        """

        if not (
            self.ocr_client.is_enabled()
            and self.ocr_config.ocr_images
            and self.ocr_budget.can_ocr_image()
        ):
            return None

        self.ocr_budget.use_image_ocr()
        result = self.ocr_client.ocr_image_file(file_path)

        if result.status != "ok":
            return None

        stat = file_path.stat()
        suffix = file_path.suffix.lower()

        text = (
            f"Image File: {file_path.name}\n"
            f"File Type: {suffix.lstrip('.')}\n"
            f"File Size Bytes: {stat.st_size}\n"
            "OCR Status: Extracted\n"
            "OCR Text:\n"
            f"{result.text}"
        )

        metadata = self._base_metadata(file_path)
        metadata.update(
            {
                "ocr_status": "extracted",
                "ocr_generated": True,
                "ocr_provider": self.ocr_config.provider,
                "ocr_languages": self.ocr_config.languages,
            }
        )

        return LoadedRecord(
            text=text,
            record_type="image_ocr_text",
            title=file_path.name,
            metadata=metadata,
        )

    def _try_caption(self, file_path: Path) -> LoadedRecord | None:
        """
        Try local vision captioning after OCR.
        """

        if not (
            self.vision_client.is_enabled()
            and self.vision_config.caption_images
            and self.caption_budget.can_caption_image()
        ):
            return None

        self.caption_budget.use_image_caption()
        result = self.vision_client.caption_image_file(file_path)

        if result.status != "ok":
            return None

        stat = file_path.stat()
        suffix = file_path.suffix.lower()

        text = (
            f"Image File: {file_path.name}\n"
            f"File Type: {suffix.lstrip('.')}\n"
            f"File Size Bytes: {stat.st_size}\n"
            "Visual Analysis Status: Captioned\n"
            "Caption Generated: True\n"
            f"Image Caption:\n{result.caption}"
        )

        metadata = self._base_metadata(file_path)
        metadata.update(
            {
                "visual_analysis_status": "captioned",
                "caption_generated": True,
                "vision_provider": self.vision_config.provider,
                "vision_model": self.vision_config.model,
            }
        )

        return LoadedRecord(
            text=text,
            record_type="image_caption",
            title=file_path.name,
            metadata=metadata,
        )

    def _build_catalog_record(self, file_path: Path) -> LoadedRecord:
        """
        Metadata-only fallback.
        """

        stat = file_path.stat()
        suffix = file_path.suffix.lower()

        text = (
            f"Image File: {file_path.name}\n"
            f"File Type: {suffix.lstrip('.')}\n"
            f"File Size Bytes: {stat.st_size}\n"
            "OCR Status: Not extracted\n"
            "Visual Analysis Status: Not enabled or not used\n"
            "Caption Generated: False\n"
            "Note: This image has been catalogued as a source artifact."
        )

        metadata = self._base_metadata(file_path)
        metadata.update(
            {
                "ocr_generated": False,
                "caption_generated": False,
                "visual_analysis_status": "not_enabled_or_not_used",
            }
        )

        return LoadedRecord(
            text=text,
            record_type="image_metadata",
            title=file_path.name,
            metadata=metadata,
        )

    @staticmethod
    def _base_metadata(file_path: Path) -> dict:
        stat = file_path.stat()
        suffix = file_path.suffix.lower()

        return {
            "file_name": file_path.name,
            "file_suffix": suffix,
            "file_size_bytes": stat.st_size,
            "loader": "ImageCatalogLoader",
        }