"""Loader contracts.

Purpose
-------
Defines the shared shapes and interface every file loader follows.

What it does
------------
``LoadedRecord`` and ``LoadedFile`` are the normalized output any loader produces;
``FileLoader`` is the protocol (supports + load) that all loaders implement, so the
pipeline can treat every format uniformly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class LoadedRecord:
    """
    One logical record extracted from a source file.

    Attributes:
        text:
            Searchable text representation of the record.

        record_type:
            Type of extracted record, such as:
                - text_document
                - csv_row
                - future: pdf_page, docx_section, xlsx_row

        title:
            Human-readable title for inspection/citation.

        metadata:
            Record-level metadata, such as row number, sheet name, page number,
            original columns, encoding, or raw row data.
    """

    text: str
    record_type: str
    title: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedFile:
    """
    Normalized loader output for any supported local file.
    """

    source_path: Path
    records: list[LoadedRecord]
    document_metadata: dict = field(default_factory=dict)


class FileLoader(Protocol):
    """
    Protocol all file loaders must follow.
    """

    supported_extensions: set[str]

    def supports(self, path: str | Path) -> bool:
        """
        Return True if this loader can process the file.
        """
        ...

    def load(self, path: str | Path) -> LoadedFile:
        """
        Load a file and return normalized records.
        """
        ...