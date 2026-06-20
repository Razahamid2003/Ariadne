"""Retrieval data models.

Purpose
-------
Defines the lightweight objects the retrieval layer passes around: the search
request, a retrieval candidate, and the search response.

What it does
------------
``RetrievalCandidate`` carries a chunk's scores, source metadata, and the reasons it
was matched; the request and response give the answer layer a clean, typed contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HybridSearchRequest:
    """
    Input for hybrid retrieval.

    query:
        Natural-language query or exact lookup query.

    top_k:
        Number of final evidence chunks to return.

    source_system:
        Optional source filter such as DOCS, IMAGES, MEALS, HEADS, GEAR, HR.

    record_type:
        Optional record-type filter such as pdf_page, csv_row, image_caption.

    target_document_types:
        Optional list of document types the query targets (from query-intent
        classification). Applied as a SOFT metadata nudge in fusion, never a
        hard filter. Empty or ['any'] means no nudge.
    """

    query: str
    top_k: int = 8
    source_system: str | None = None
    record_type: str | None = None
    target_document_types: tuple[str, ...] = ()


@dataclass
class RetrievalCandidate:
    """
    One retrieved evidence candidate.

    The same chunk may be found by vector search, keyword search, or both. The
    hybrid retriever merges by chunk_id and stores each score separately.
    """

    chunk_id: str
    document_id: str
    source_system: str
    source_file: str
    record_type: str
    title: str
    citation_label: str
    text: str
    metadata_json: str | None = None

    vector_score: float = 0.0
    keyword_score: float = 0.0
    exact_match_score: float = 0.0
    reranker_score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None
    record_type_weight: float = 1.0
    combined_score: float = 0.0
    match_reasons: list[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        """
        Add a match reason without duplicates.
        """

        if reason and reason not in self.match_reasons:
            self.match_reasons.append(reason)

    def to_dict(self, preview_chars: int = 600) -> dict[str, Any]:
        """
        Convert to JSON-friendly dict.
        """

        preview = self.text[:preview_chars] if self.text else ""

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_system": self.source_system,
            "source_file": self.source_file,
            "record_type": self.record_type,
            "title": self.title,
            "citation_label": self.citation_label,
            "vector_score": round(self.vector_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "exact_match_score": round(self.exact_match_score, 4),
            "reranker_score": round(self.reranker_score, 4),
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "record_type_weight": round(self.record_type_weight, 4),
            "combined_score": round(self.combined_score, 4),
            "match_reasons": self.match_reasons,
            "text_preview": preview,
        }


@dataclass(frozen=True)
class HybridSearchResponse:
    """
    Final hybrid retrieval response.
    """

    query: str
    confidence: str
    results: list[RetrievalCandidate]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, preview_chars: int = 600) -> dict[str, Any]:
        """
        Convert to JSON-friendly dict.
        """

        return {
            "query": self.query,
            "confidence": self.confidence,
            "result_count": len(self.results),
            "diagnostics": self.diagnostics,
            "results": [
                {"rank": index + 1, **candidate.to_dict(preview_chars=preview_chars)}
                for index, candidate in enumerate(self.results)
            ],
        }
