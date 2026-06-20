"""Shared ingestion data schemas.

Purpose
-------
Defines the two core records that flow through ingestion and indexing: a source
document and a searchable, citable chunk.

What it does
------------
``DocumentRecord`` represents one ingested source file or logical document;
``ChunkRecord`` represents one retrievable chunk and can produce its stable
citation label.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """
    Represents one ingested source file or logical source document.

    Examples:
        - A Markdown maintenance note
        - A PDF manual
        - A DOCX technical report
        - A CSV export from MEALS
        - An XLSX export from HEADS
    """

    document_id: str
    source_file: str
    source_system: str = "documents"
    record_type: str = "document"
    title: str | None = None
    sensitivity: str = "internal"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(BaseModel):
    """
    Represents one searchable/citable chunk.

    Chunks are the unit that will later be:
        - embedded
        - indexed
        - retrieved
        - passed into RAG context
        - cited in generated answers
    """

    chunk_id: str
    document_id: str
    text: str
    source_file: str
    source_system: str = "documents"
    record_type: str = "chunk"
    title: str | None = None
    chunk_index: int
    sensitivity: str = "internal"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation_label(self) -> str:
        """
        Return a stable citation label for this chunk.

        Example:
            [MEALS: doc-machine-554-a1b2c3d4e5-chunk-0000]
        """

        return f"[{self.source_system}: {self.chunk_id}]"