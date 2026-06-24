"""Table-aware retrieval support.

Some questions require reasoning over an entire table rather than a few rows:
counting ("how many are backordered"), summing ("total quantity"), or finding an
extreme ("longest lead time"). Ordinary top-k retrieval surfaces only a handful
of rows, so the model literally cannot see the whole table and under-counts or
declines.

This module addresses that WITHOUT computing answers or biasing ranking:

  1. ``is_aggregation_query`` detects, from the wording alone, that a question is
     about counting / totalling / enumerating / extremes. It is generic language
     detection — it knows nothing about any particular corpus, document, or answer.

  2. ``complete_tables`` looks at the tables the retriever ALREADY surfaced and
     loads their remaining rows from storage so the model can see the full table.
     The added rows are appended after the ranked results at a neutral score (the
     lowest score the table's own retrieved rows already earned) — they are never
     ranked above genuine hits, and no row is singled out or boosted.

The model still does all the counting and reasoning itself; this only makes the
evidence complete.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.retrieval.models import RetrievalCandidate

# Generic aggregation / enumeration / extreme-value language. No domain terms.
_AGG_PATTERNS = [
    r"\bhow many\b", r"\bnumber of\b", r"\bcount\b", r"\bcounting\b",
    r"\btotal\b", r"\bsum\b", r"\baltogether\b", r"\bin total\b", r"\bcombined\b",
    r"\blist (all|the|every|each|out)\b", r"\bname (all|every|the)\b",
    r"\ball (of )?(the )?\w+", r"\bevery\b", r"\beach\b",
    r"\bmost\b", r"\bleast\b", r"\bfewest\b", r"\bhighest\b", r"\blowest\b",
    r"\blargest\b", r"\bsmallest\b", r"\blongest\b", r"\bshortest\b",
    r"\bgreatest\b", r"\bbiggest\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bmax\b", r"\bmin\b", r"\baverage\b", r"\bmean\b", r"\bmedian\b",
    r"\bper (item|unit|row|supplier|category|type|variant)\b", r"\bacross all\b",
    r"\bwhich .*\b(are|have|has|were|is)\b",
]
_AGG_RE = re.compile("|".join(_AGG_PATTERNS), re.IGNORECASE)

# A tabular row chunk is tagged by the loaders with a record_type ending in "_row"
# (e.g. "csv_row", "xlsx_row"). This is the generic signal — not a filename.
def is_table_row(record_type: str | None) -> bool:
    return bool(record_type) and str(record_type).endswith("_row")


def is_aggregation_query(query: str) -> bool:
    """True if the wording indicates counting / totalling / enumeration / extremes."""
    return bool(_AGG_RE.search(query or ""))


def _row_dict_to_candidate(row: dict[str, Any], score: float) -> RetrievalCandidate:
    cand = RetrievalCandidate(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        source_system=row["source_system"],
        source_file=row["source_file"],
        record_type=row["record_type"],
        title=row.get("title") or "",
        citation_label=row["citation_label"],
        text=row["text"],
        metadata_json=row.get("metadata_json"),
        combined_score=score,
    )
    cand.add_reason("table_completion")
    return cand


def complete_tables(
    results: list[RetrievalCandidate],
    store: Any,
    max_rows: int = 400,
) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
    """Append the remaining rows of any table already present in ``results``.

    Returns (expanded_results, diagnostics). Does not reorder or rescore the
    existing results, and adds rows only for tables the retriever already
    surfaced.
    """

    # Which documents are tables, and what is the lowest score their already-
    # retrieved rows earned? We reuse that score for completion rows so they are
    # neither boosted above nor buried below the table's demonstrated relevance.
    table_docs: dict[str, float] = {}
    present_chunk_ids: set[str] = set()
    for c in results:
        present_chunk_ids.add(c.chunk_id)
        if is_table_row(c.record_type):
            score = c.combined_score
            if c.document_id not in table_docs or score < table_docs[c.document_id]:
                table_docs[c.document_id] = score

    if not table_docs:
        return results, {"table_completion_applied": False}

    added: list[RetrievalCandidate] = []
    completed: list[str] = []
    budget = max_rows
    for document_id, neutral_score in table_docs.items():
        if budget <= 0:
            break
        try:
            rows = store.list_chunks_for_document(document_id)
        except Exception:
            continue
        rows = [r for r in rows if is_table_row(r.get("record_type"))]
        new_rows = [r for r in rows if r["chunk_id"] not in present_chunk_ids]
        if not new_rows:
            continue
        take = new_rows[: max(0, budget)]
        for r in take:
            added.append(_row_dict_to_candidate(r, neutral_score))
            present_chunk_ids.add(r["chunk_id"])
        budget -= len(take)
        completed.append(document_id)

    diagnostics = {
        "table_completion_applied": bool(added),
        "tables_completed": len(completed),
        "rows_added": len(added),
    }
    # Append after the ranked results; do not re-sort the ranked portion.
    return results + added, diagnostics
