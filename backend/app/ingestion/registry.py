"""Loader registry.

Purpose
-------
Maps file extensions to the loader that handles them, so the pipeline can find the
right loader for any input file.

What it does
------------
Lists supported extensions, returns the loader for a given file, and reports
whether a file type is supported.
"""

from pathlib import Path

from backend.app.core.config import OcrConfig, VisionConfig
from backend.app.ingestion.loaders.base import FileLoader
from backend.app.ingestion.loaders.document_file_loader import DocumentFileLoader
from backend.app.ingestion.loaders.image_catalog_loader import ImageCatalogLoader
from backend.app.ingestion.loaders.local_file_loader import LocalFileLoader
from backend.app.ocr.ocr_budget import OcrBudget
from backend.app.vision.caption_budget import CaptionBudget


class LoaderRegistry:
    """
    Registry of available file loaders.
    """

    def __init__(
        self,
        loaders: list[FileLoader] | None = None,
        vision_config: VisionConfig | None = None,
        ocr_config: OcrConfig | None = None,
    ):
        active_vision_config = vision_config or VisionConfig()
        active_ocr_config = ocr_config or OcrConfig()

        caption_budget = CaptionBudget.from_config(active_vision_config)
        ocr_budget = OcrBudget.from_config(active_ocr_config)

        self.loaders = loaders or [
            LocalFileLoader(),
            DocumentFileLoader(
                vision_config=active_vision_config,
                caption_budget=caption_budget,
                ocr_config=active_ocr_config,
                ocr_budget=ocr_budget,
            ),
            ImageCatalogLoader(
                vision_config=active_vision_config,
                caption_budget=caption_budget,
                ocr_config=active_ocr_config,
                ocr_budget=ocr_budget,
            ),
        ]

    def supported_extensions(self) -> set[str]:
        extensions: set[str] = set()

        for loader in self.loaders:
            extensions.update(loader.supported_extensions)

        return extensions

    def get_loader(self, path: str | Path) -> FileLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader

        return None

    def is_supported(self, path: str | Path) -> bool:
        return self.get_loader(path) is not None