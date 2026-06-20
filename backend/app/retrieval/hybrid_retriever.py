"""Hybrid retriever.

Purpose
-------
The main retrieval entry point: given a query, it returns the best evidence by
combining semantic and keyword search.

What it does
------------
Runs dense vector search and keyword search, merges and de-duplicates the results,
fuses their scores, applies the document-type signal, optionally reranks the top
pool, and returns ranked candidates with a confidence estimate.

Flow
----
query -> embed and vector-search + keyword-search -> merge by chunk and remove
duplicates -> fuse scores -> nudge by document type -> rerank top pool ->
return the ranked evidence and confidence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from backend.app.retrieval.fusion import apply_document_type_nudge, estimate_confidence, fuse_candidates
from backend.app.retrieval.query_normalizer import normalize_query
from backend.app.retrieval.models import (
    HybridSearchRequest,
    HybridSearchResponse,
    RetrievalCandidate,
)

PAGE_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


class HybridRetriever:
    """Combine dense vector retrieval and keyword retrieval via rank fusion.

    Heavy collaborators (embedding model, vector index, keyword index,
    cross-encoder) are built lazily on first use. This keeps the module
    importable and its fusion logic testable in environments without the model
    runtime installed, and avoids paying model-load cost until a query arrives.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._keyword_index: Any | None = None
        self._vector_index: Any | None = None
        self._embedding_model: Any | None = None
        self._reranker: Any | None = None

    # ------------------------------------------------------------------ #
    # Lazy collaborators
    # ------------------------------------------------------------------ #
    @property
    def keyword_index(self) -> Any:
        if self._keyword_index is None:
            from backend.app.retrieval.keyword_index import KeywordSearchIndex

            self._keyword_index = KeywordSearchIndex(
                metadata_db_path=self.settings.paths.metadata_db,
                table_name=getattr(self.settings.retrieval, "keyword_table", "chunks_fts"),
            )
        return self._keyword_index

    @property
    def vector_index(self) -> Any:
        if self._vector_index is None:
            from backend.app.indexing.vector_index import LocalVectorIndex

            self._vector_index = LocalVectorIndex(
                index_dir=self.settings.vector_index.index_dir,
                embeddings_file=self.settings.vector_index.embeddings_file,
                metadata_file=self.settings.vector_index.metadata_file,
            )
        return self._vector_index

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            from backend.app.embeddings.local_embedding_model import (
                SentenceTransformersEmbeddingModel,
            )

            self._embedding_model = SentenceTransformersEmbeddingModel(self.settings.embeddings)
        return self._embedding_model

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            from backend.app.retrieval.reranker import LocalReranker

            self._reranker = LocalReranker(self.settings)
        return self._reranker

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(self, request: HybridSearchRequest) -> HybridSearchResponse:
        query = (request.query or "").strip()
        cfg = self.settings.retrieval
        query = normalize_query(query, getattr(cfg, "query_expansions", None))
        if not query:
            return HybridSearchResponse(
                query=request.query,
                confidence="low",
                results=[],
                diagnostics={"error": "Empty query."},
            )
        vector_top_k = max(request.top_k, cfg.vector_top_k)
        keyword_top_k = max(request.top_k, cfg.keyword_top_k)
        final_top_k = request.top_k or cfg.final_top_k

        vector_candidates = self._vector_search(
            query=query,
            top_k=vector_top_k,
            source_system=request.source_system,
            record_type=request.record_type,
        )
        keyword_candidates = self._keyword_search(
            query=query,
            top_k=keyword_top_k,
            source_system=request.source_system,
            record_type=request.record_type,
        )

        merged = self._merge(vector_candidates, keyword_candidates)

        fuse_candidates(
            list(merged.values()),
            method=cfg.fusion_method,
            rrf_k=cfg.rrf_k,
            vector_weight=cfg.vector_weight,
            keyword_weight=cfg.keyword_weight,
            record_type_weights=getattr(cfg, "record_type_weights", None),
            apply_record_type_prior=bool(getattr(cfg, "record_type_weights", None)),
        )

        # Metadata leg of retrieval: soft, transparent
        # nudge for candidates whose document_type matches the query intent.
        if request.target_document_types:
            apply_document_type_nudge(
                list(merged.values()),
                list(request.target_document_types),
                nudge_weight=float(getattr(cfg, "document_type_nudge_weight", 0.15)),
            )

        ranked = sorted(merged.values(), key=lambda item: item.combined_score, reverse=True)

        deduped = self._deduplicate(ranked) if cfg.deduplicate else ranked

        # Rerank a pool slightly larger than final_top_k so the cross-encoder can
        # promote a strong-but-lower-fused candidate into the final set.
        rerank_pool = max(final_top_k, int(getattr(cfg, "rerank_candidate_pool", 12)))
        reranked, rerank_diag = self.reranker.rerank(query, deduped, top_k=rerank_pool)

        filtered = [c for c in reranked if c.combined_score >= cfg.min_score]
        # Never let the min_score filter empty an otherwise-non-empty result set;
        # fall back to the best available so the RAG layer can still decide.
        if not filtered and reranked:
            filtered = reranked[: min(final_top_k, len(reranked))]

        final_results = filtered[:final_top_k]
        confidence = estimate_confidence(
            final_results,
            rerank_used=bool(rerank_diag.get("reranker_used")),
            high_score=getattr(cfg, "confidence_high_score", 0.55),
            medium_score=getattr(cfg, "confidence_medium_score", 0.35),
        )

        diagnostics = {
            "vector_candidates": len(vector_candidates),
            "keyword_candidates": len(keyword_candidates),
            "merged_candidates": len(merged),
            "after_deduplication": len(deduped),
            "after_min_score": len(filtered),
            "returned": len(final_results),
            "fusion_method": cfg.fusion_method,
            "rrf_k": cfg.rrf_k,
            "vector_weight": cfg.vector_weight,
            "keyword_weight": cfg.keyword_weight,
            "min_score": cfg.min_score,
            **rerank_diag,
        }

        return HybridSearchResponse(
            query=query,
            confidence=confidence,
            results=final_results,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Retriever calls
    # ------------------------------------------------------------------ #
    def _vector_search(
        self,
        query: str,
        top_k: int,
        source_system: str | None,
        record_type: str | None,
    ) -> list[RetrievalCandidate]:
        import numpy as np

        batch = self.embedding_model.encode([query], show_progress_bar=False)
        vector = np.asarray(batch.vectors)
        if vector.ndim != 2 or vector.shape[0] != 1:
            vector = vector.reshape(1, -1)

        results = self.vector_index.search(
            query_vector=vector,
            top_k=top_k,
            source_system=source_system,
            record_type=record_type,
        )

        candidates: list[RetrievalCandidate] = []
        for rank, result in enumerate(results, start=1):
            candidate = self._result_to_candidate(result)
            candidate.vector_score = float(self._get(result, "score", 0.0) or 0.0)
            candidate.vector_rank = rank
            candidate.add_reason("vector")
            candidates.append(candidate)
        return candidates

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        source_system: str | None,
        record_type: str | None,
    ) -> list[RetrievalCandidate]:
        results = self.keyword_index.search(
            query=query,
            top_k=top_k,
            source_system=source_system,
            record_type=record_type,
        )
        for rank, candidate in enumerate(results, start=1):
            candidate.keyword_rank = rank
            candidate.add_reason("keyword")
        return results

    # ------------------------------------------------------------------ #
    # Merge / dedup helpers (domain-agnostic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(
        vector_candidates: list[RetrievalCandidate],
        keyword_candidates: list[RetrievalCandidate],
    ) -> dict[str, RetrievalCandidate]:
        merged: dict[str, RetrievalCandidate] = {}
        for candidate in vector_candidates:
            merged[candidate.chunk_id] = candidate
        for candidate in keyword_candidates:
            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
                continue
            existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
            existing.keyword_rank = candidate.keyword_rank
            for reason in candidate.match_reasons:
                existing.add_reason(reason)
            if not existing.text and candidate.text:
                existing.text = candidate.text
            if not existing.metadata_json and candidate.metadata_json:
                existing.metadata_json = candidate.metadata_json
        return merged

    @classmethod
    def _deduplicate(cls, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        """Suppress obvious duplicates (same file+page+format group, or same
        file+format+near-identical text). Conservative and domain-agnostic:
        OCR / caption / native variants of the same page are kept separate
        because they may carry different evidence."""

        kept: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = cls._dedup_key(candidate)
            if key in seen:
                candidate.add_reason("duplicate_suppressed")
                continue
            seen.add(key)
            kept.append(candidate)
        return kept

    @staticmethod
    def _dedup_key(candidate: RetrievalCandidate) -> str:
        source_name = Path(candidate.source_file).name.lower().strip()
        title = candidate.title.lower().strip()
        record_type = candidate.record_type.lower().strip()
        text_sample = " ".join((candidate.text or "").lower().split())[:500]
        group = HybridRetriever._record_group(record_type)

        page_match = PAGE_PATTERN.search(title)
        if page_match and source_name:
            return f"pdf::{source_name}::page::{page_match.group(1)}::{group}"
        if source_name:
            return f"file::{source_name}::{group}::{HybridRetriever._hash_text(text_sample)}"
        return f"chunk::{candidate.chunk_id}"

    @staticmethod
    def _record_group(record_type: str) -> str:
        if record_type in {"pdf_page", "csv_row", "xlsx_row", "docx_section", "pptx_slide", "text_document"}:
            return "native"
        if "ocr" in record_type:
            return "ocr"
        if "caption" in record_type:
            return "vision"
        if "metadata" in record_type:
            return "metadata"
        return record_type or "unknown"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    # Result mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get(result: Any, name: str, default: Any = "") -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    @classmethod
    def _result_to_candidate(cls, result: Any) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=str(cls._get(result, "chunk_id")),
            document_id=str(cls._get(result, "document_id", "")),
            source_system=str(cls._get(result, "source_system", "")),
            source_file=str(cls._get(result, "source_file", "")),
            record_type=str(cls._get(result, "record_type", "")),
            title=str(cls._get(result, "title", "")),
            citation_label=str(cls._get(result, "citation_label", "")),
            text=str(cls._get(result, "text", "")),
            metadata_json=str(cls._get(result, "metadata_json", "")),
        )
