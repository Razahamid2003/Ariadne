"""Evidence context builder.

Purpose
-------
Turns ranked retrieval results into a compact, clean, citation-ready context for
the model, within a predictable size budget.

What it does
------------
Selects the strongest candidates, cleans their text (normalization, fixing broken
OCR lines, trimming fragments), optionally adds neighboring context, and assembles
the evidence packet the prompt will use.

Flow
----
Candidates are filtered and de-noised, each becomes a labeled evidence item, and
the packet is capped to a character and chunk budget so the prompt stays within the
model's context window.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata

from backend.app.core.config import Settings
from backend.app.rag.models import BuiltRAGContext, EvidenceChunk
from backend.app.retrieval.models import HybridSearchResponse, RetrievalCandidate

STRUCTURED_RECORD_TYPES = {"csv_row", "xlsx_row"}

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RANGE_PATTERN = re.compile(
    rf"\b{_MONTH}\s+\d{{4}}\s*(?:-|–|—|to|through)\s*(?:Present|Current|Ongoing|{_MONTH}\s+\d{{4}})\b",
    re.IGNORECASE,
)
YEAR_RANGE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\s*(?:-|–|—|to|through)\s*(?:present|current|ongoing|(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


class RAGContextBuilder:
    """Build the evidence context used by the answer-generation prompt.

    The builder trusts retrieval order. It performs only generic evidence
    hygiene: text cleanup, OCR repair, row/section formatting, optional
    neighbor context, and budget enforcement.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build(self, retrieval_response: HybridSearchResponse) -> BuiltRAGContext:
        from backend.app.retrieval.aggregation import is_table_row

        max_chunks = self.settings.rag.max_context_chunks
        max_chars = self.settings.rag.max_context_chars
        per_chunk_chars = self.settings.rag.max_chars_per_chunk

        # When the retriever has completed a table (aggregation query), allow a
        # larger character budget so the whole table can fit, and let table rows
        # past the chunk-count cap. Table rows are small; the char budget still
        # bounds total context. Non-table behaviour is unchanged.
        has_table = any(
            "table_completion" in getattr(c, "match_reasons", []) or is_table_row(c.record_type)
            for c in retrieval_response.results
        )
        if has_table:
            max_chars = max(max_chars, int(getattr(self.settings.rag, "max_context_chars_table", 24000)))

        evidence: list[EvidenceChunk] = []
        blocks: list[str] = []
        used_chars = 0
        truncated = False
        seen_chunk_ids: set[str] = set()
        neighbor_context_added = 0
        dirty_spans_repaired = 0

        def add_candidate(candidate: RetrievalCandidate, neighbor: bool = False) -> bool:
            nonlocal used_chars, truncated, neighbor_context_added, dirty_spans_repaired
            # Table rows are exempt from the chunk-count cap (so a complete table
            # is not cut off mid-way); the character budget below still bounds them.
            if len(evidence) >= max_chunks and not is_table_row(candidate.record_type):
                truncated = True
                return False
            if candidate.chunk_id in seen_chunk_ids:
                return False
            if self._should_skip_candidate(candidate):
                return False

            prepared_text, repaired = self._prepare_text(candidate, max_chars=per_chunk_chars)
            if repaired:
                dirty_spans_repaired += 1
            if not prepared_text:
                return False

            block = self._format_evidence_block(len(evidence) + 1, candidate, prepared_text)
            block_len = len(block)
            if used_chars + block_len > max_chars:
                remaining = max_chars - used_chars
                if remaining <= 500:
                    truncated = True
                    return False
                prepared_text = self._truncate_cleanly(prepared_text, max(0, remaining - 500), candidate.record_type)
                block = self._format_evidence_block(len(evidence) + 1, candidate, prepared_text)
                block_len = len(block)
                truncated = True

            seen_chunk_ids.add(candidate.chunk_id)
            evidence.append(EvidenceChunk.from_candidate(len(evidence) + 1, candidate, prepared_text))
            blocks.append(block)
            used_chars += block_len
            if neighbor:
                neighbor_context_added += 1
            return True

        for candidate in retrieval_response.results:
            # Do not stop early: add_candidate enforces the chunk cap for non-table
            # candidates and the char budget for all, while letting table rows
            # through so a completed table is not cut short.
            added = add_candidate(candidate, neighbor=False)
            if added and self._supports_neighbor_context(candidate):
                for neighbor in self._neighbor_candidates(candidate, seen_chunk_ids):
                    if len(evidence) >= max_chunks:
                        truncated = True
                        break
                    add_candidate(neighbor, neighbor=True)

        context_text = "\n\n".join(blocks)
        diagnostics = {
            "retrieved_results": len(retrieval_response.results),
            "context_chunks": len(evidence),
            "filtered_out_before_context": max(0, len(retrieval_response.results) - len(evidence)),
            "max_context_chunks": max_chunks,
            "max_context_chars": max_chars,
            "max_chars_per_chunk": per_chunk_chars,
            "neighbor_context_added": neighbor_context_added,
            "dirty_spans_repaired": dirty_spans_repaired,
            "truncated": truncated,
        }
        return BuiltRAGContext(
            query=retrieval_response.query,
            retrieval_confidence=retrieval_response.confidence,
            evidence=evidence,
            context_text=context_text,
            total_chars=len(context_text),
            truncated=truncated,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Candidate gating (generic, score-based only)
    # ------------------------------------------------------------------ #
    def _should_skip_candidate(self, candidate: RetrievalCandidate) -> bool:
        """Drop weak vector-only chunks below a configurable score.

        Uses only match-reason provenance ("vector" present, "keyword" absent)
        and the normalized combined_score. No content/category inspection.
        """

        threshold = getattr(self.settings.rag, "drop_vector_only_below_score", 0.0)
        if threshold <= 0:
            return False
        reasons = set(candidate.match_reasons or [])
        is_vector_only = "vector" in reasons and "keyword" not in reasons
        return bool(is_vector_only and candidate.combined_score < threshold)

    # ------------------------------------------------------------------ #
    # Neighbor context
    # ------------------------------------------------------------------ #
    @staticmethod
    def _supports_neighbor_context(candidate: RetrievalCandidate) -> bool:
        record_type = (candidate.record_type or "").lower()
        if record_type in STRUCTURED_RECORD_TYPES:
            return False
        return bool(candidate.document_id and candidate.chunk_id)

    def _neighbor_candidates(self, candidate: RetrievalCandidate, seen_chunk_ids: set[str]) -> list[RetrievalCandidate]:
        try:
            with sqlite3.connect(self.settings.paths.metadata_db) as conn:
                conn.row_factory = sqlite3.Row
                current = conn.execute(
                    "SELECT chunk_index FROM chunks WHERE chunk_id = ? LIMIT 1;",
                    (candidate.chunk_id,),
                ).fetchone()
                if current is None:
                    return []
                chunk_index = int(current["chunk_index"])
                rows = conn.execute(
                    """
                    SELECT chunk_id, document_id, text, source_file, source_system,
                           record_type, title, citation_label, metadata_json, chunk_index
                    FROM chunks
                    WHERE document_id = ?
                      AND chunk_index BETWEEN ? AND ?
                      AND chunk_id <> ?
                    ORDER BY ABS(chunk_index - ?), chunk_index ASC
                    LIMIT 2;
                    """,
                    (candidate.document_id, chunk_index - 1, chunk_index + 1, candidate.chunk_id, chunk_index),
                ).fetchall()
        except Exception:
            return []

        neighbors: list[RetrievalCandidate] = []
        for row in rows:
            chunk_id = str(row["chunk_id"] or "")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            neighbors.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    document_id=str(row["document_id"] or ""),
                    source_system=str(row["source_system"] or ""),
                    source_file=str(row["source_file"] or ""),
                    record_type=str(row["record_type"] or ""),
                    title=str(row["title"] or ""),
                    citation_label=str(row["citation_label"] or ""),
                    text=str(row["text"] or ""),
                    metadata_json=str(row["metadata_json"] or ""),
                    combined_score=max(candidate.combined_score * 0.96, candidate.combined_score - 0.01),
                    match_reasons=["neighbor_context"],
                )
            )
        return neighbors

    # ------------------------------------------------------------------ #
    # Text preparation / hygiene
    # ------------------------------------------------------------------ #
    @classmethod
    def _prepare_text(cls, candidate: RetrievalCandidate, max_chars: int) -> tuple[str, bool]:
        raw = candidate.text or ""
        if (candidate.record_type or "").lower() in STRUCTURED_RECORD_TYPES:
            text = cls._clean_structured_record(raw)
            return cls._truncate_cleanly(text, max_chars, candidate.record_type), False
        before = raw
        text = cls._clean_document_text(raw)
        text, repaired = cls._repair_dirty_span(text)
        text = cls._truncate_cleanly(text, max_chars, candidate.record_type)
        return text, repaired or (text != before.strip())

    @staticmethod
    def _clean_structured_record(text: str | None) -> str:
        if not text:
            return ""
        value = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
        lines = [re.sub(r"\s+", " ", line).strip() for line in value.split("\n")]
        return "\n".join(line for line in lines if line).strip()

    @classmethod
    def _clean_document_text(cls, text: str | None) -> str:
        if not text:
            return ""
        value = unicodedata.normalize("NFKC", str(text)).replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)
        value = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1-\2", value)  # de-hyphenate across line breaks
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        lines: list[str] = []
        for line in value.split("\n"):
            cleaned_line = re.sub(r"\s+", " ", line).strip()
            if not cleaned_line:
                continue
            if cls._is_broken_ocr_line(cleaned_line):
                continue
            lines.append(cleaned_line)
        return "\n".join(lines).strip()

    @classmethod
    def _repair_dirty_span(cls, text: str) -> tuple[str, bool]:
        if not text:
            return "", False
        repaired = False
        value = text.strip()
        if cls._starts_like_fragment(value):
            next_start = cls._next_clean_start(value)
            if next_start is not None and next_start < min(len(value), 1200):
                value = value[next_start:].lstrip(" -–—:;,.\n\t")
                repaired = True
            else:
                return "", True
        lines = value.split("\n")
        while len(lines) > 1 and cls._looks_like_orphan_prefix(lines[0]):
            lines.pop(0)
            repaired = True
        value = "\n".join(lines).strip()
        if value and cls._starts_like_fragment(value):
            return "", True
        return value, repaired

    @staticmethod
    def _is_broken_ocr_line(line: str) -> bool:
        value = (line or "").strip()
        if not value:
            return False
        if re.match(r"^[a-z]{1,2}\s+(?:by|and|or|of|to|in|with)\b", value):
            return True
        if len(value) < 32 and value[0].islower() and not value.endswith((".", ":", ";", ")")):
            return True
        return False

    @staticmethod
    def _starts_like_fragment(text: str) -> bool:
        if not text:
            return False
        first = text[:120].lstrip()
        if not first:
            return False
        if first.startswith(("-", "•", "*", "#", "|")):
            return False
        if re.match(r"^(?:[A-Z][a-z]+\s+){1,6}(?:-|–|—|:)\s", first):
            return False
        if first[0].islower():
            return True
        if re.match(r"^[A-Za-z]{1,2}\s+(?:by|and|or|of|to|in|with)\b", first):
            return True
        return False

    @staticmethod
    def _looks_like_orphan_prefix(line: str) -> bool:
        value = (line or "").strip()
        if not value:
            return True
        if len(value) <= 3:
            return True
        if len(value) < 32 and value[0].islower() and not value.endswith(('.', ':', ';')):
            return True
        return False

    @staticmethod
    def _next_clean_start(text: str) -> int | None:
        patterns = [
            r"\n\s*(?:[-•*]|\d+[.)])\s+",
            r"(?:^|\n)\s*[A-Z][A-Za-z0-9 /&,+()'\-]{2,80}\s*(?:-|–|—|:)\s+",
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
        ]
        starts: list[int] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                starts.append(match.end() if pattern.startswith("(?<=") else match.start())
        return min(starts) if starts else None

    @classmethod
    def _truncate_cleanly(cls, text: str, max_chars: int, record_type: str | None = None) -> str:
        if not text or max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text.strip()
        if (record_type or "").lower() in STRUCTURED_RECORD_TYPES:
            return text[:max_chars].rstrip()
        cutoff = max_chars
        window = text[:max_chars]
        boundaries = [window.rfind(marker) for marker in ("\n\n", "\n- ", "\n• ", ". ", "? ", "! ", "\n")]
        best = max(boundaries)
        if best >= max(350, int(max_chars * 0.55)):
            cutoff = best + 1
        return text[:cutoff].rstrip(" ,;:-")

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #
    @classmethod
    def _detected_date_ranges(cls, text: str) -> list[str]:
        if not text:
            return []
        matches = DATE_RANGE_PATTERN.findall(text) + YEAR_RANGE_PATTERN.findall(text)
        out: list[str] = []
        for item in matches:
            value = item if isinstance(item, str) else "".join(part for part in item if part)
            value = re.sub(r"\s+", " ", value).strip()
            if value and value.lower() not in {v.lower() for v in out}:
                out.append(value)
        return out[:5]

    @classmethod
    def _format_evidence_block(cls, evidence_index: int, candidate: RetrievalCandidate, text: str) -> str:
        reasons = ", ".join(candidate.match_reasons[:8]) if candidate.match_reasons else "retrieved"
        date_ranges = cls._detected_date_ranges(text)
        date_line = f"Detected Date Ranges: {'; '.join(date_ranges)}\n" if date_ranges else ""
        return (
            f"Evidence ID: E{evidence_index}\n"
            f"Citation Label: {candidate.citation_label}\n"
            f"Title: {candidate.title}\n"
            f"Source System: {candidate.source_system}\n"
            f"Record Type: {candidate.record_type}\n"
            f"Source File: {candidate.source_file}\n"
            f"Retrieval Score: {candidate.combined_score:.4f}\n"
            f"Match Reasons: {reasons}\n"
            f"{date_line}"
            f"Evidence Text:\n{text}"
        )
