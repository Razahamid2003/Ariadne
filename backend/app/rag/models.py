"""Answer-generation data models.

Purpose
-------
Defines the typed, JSON-friendly objects that flow through answer generation: the
request, evidence items, the built context, the validation result, and the final
response.

What it does
------------
Each model converts to a dictionary for API/CLI use; ``EvidenceChunk`` builds itself
from a retrieval candidate, and the response carries the answer, status, citations,
confidence, evidence, and diagnostics the UI reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.app.retrieval.models import RetrievalCandidate
from backend.app.rag.diagnostics import single_pass_diagnostics


@dataclass(frozen=True)
class RAGAnswerRequest:
    """
    Input for grounded answer generation.

    query:
        User question.

    top_k:
        Number of evidence chunks requested from hybrid retrieval.

    source_system / record_type:
        Optional filters passed through to hybrid retrieval.

    show_evidence:
        If true, CLI/API callers may include evidence previews in the response.
    """

    query: str
    top_k: int = 8
    source_system: str | None = None
    record_type: str | None = None
    show_evidence: bool = False
    answer_mode: str = "balanced"
    conversation_context: str = ""
    prior_evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class EvidenceChunk:
    """
    Evidence item prepared for the LLM context.

    citation_label is the only citation token the model is allowed to use.
    """

    evidence_id: str
    citation_label: str
    chunk_id: str
    document_id: str
    source_system: str
    source_file: str
    record_type: str
    title: str
    text: str
    combined_score: float
    document_type: str = ""
    match_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_candidate(cls, index: int, candidate: RetrievalCandidate, text: str) -> "EvidenceChunk":
        document_type = ""
        raw = getattr(candidate, "metadata_json", None)
        if raw:
            try:
                import json
                data = json.loads(raw) if isinstance(raw, str) else dict(raw)
                document_type = str(data.get("document_type", "") or "")
            except (json.JSONDecodeError, ValueError, TypeError):
                document_type = ""
        return cls(
            evidence_id=f"E{index}",
            citation_label=candidate.citation_label,
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            source_system=candidate.source_system,
            source_file=candidate.source_file,
            record_type=candidate.record_type,
            title=candidate.title,
            text=text,
            combined_score=candidate.combined_score,
            document_type=document_type,
            match_reasons=list(candidate.match_reasons),
        )

    def to_dict(self, preview_chars: int = 700) -> dict[str, Any]:
        preview = self.text[:preview_chars] if self.text else ""
        return {
            "evidence_id": self.evidence_id,
            "citation_label": self.citation_label,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_system": self.source_system,
            "source_file": self.source_file,
            "record_type": self.record_type,
            "title": self.title,
            "combined_score": round(self.combined_score, 4),
            "match_reasons": self.match_reasons,
            "text_preview": preview,
        }


@dataclass(frozen=True)
class BuiltRAGContext:
    """
    Final context packet supplied to the prompt builder.
    """

    query: str
    retrieval_confidence: str
    evidence: list[EvidenceChunk]
    context_text: str
    total_chars: int
    truncated: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_labels(self) -> list[str]:
        return [item.citation_label for item in self.evidence]

    def to_dict(self, preview_chars: int = 700) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_confidence": self.retrieval_confidence,
            "evidence_count": len(self.evidence),
            "total_chars": self.total_chars,
            "truncated": self.truncated,
            "diagnostics": self.diagnostics,
            "evidence": [item.to_dict(preview_chars=preview_chars) for item in self.evidence],
        }


@dataclass(frozen=True)
class CitationValidationResult:
    """
    Result of citation validation over the generated answer.
    """

    valid: bool
    cited_labels: list[str]
    unused_evidence_labels: list[str]
    unexpected_labels: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "cited_labels": self.cited_labels,
            "unused_evidence_labels": self.unused_evidence_labels,
            "unexpected_labels": self.unexpected_labels,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RAGAnswerResponse:
    """
    Final response returned by the answer generator.
    """

    query: str
    answer: str
    status: str
    confidence: str
    model: str | None
    citations: list[str]
    validation: CitationValidationResult
    retrieval_diagnostics: dict[str, Any]
    evidence: list[EvidenceChunk]
    llm_latency_ms: int | None = None
    error: str | None = None
    used_retry: bool = False

    def to_dict(self, preview_chars: int = 700, include_evidence: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "status": self.status,
            "confidence": self.confidence,
            "model": self.model,
            "answer": self.answer,
            "citations": self.citations,
            "validation": self.validation.to_dict(),
            "retrieval_diagnostics": single_pass_diagnostics(self.retrieval_diagnostics),
            "llm_latency_ms": self.llm_latency_ms,
            "used_retry": self.used_retry,
            "error": self.error,
        }

        if include_evidence:
            payload["evidence"] = [item.to_dict(preview_chars=preview_chars) for item in self.evidence]

        return payload
