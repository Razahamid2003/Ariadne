"""Keyword search index.

Purpose
-------
Provides fast exact and keyword search over the chunk text using SQLite's built-in
full-text search, so precise terms and IDs are found reliably.

What it does
------------
Builds a full-text index over the chunks table, runs keyword queries, scores
matches, and falls back gracefully when full-text search is unavailable.

Flow
----
``rebuild()`` populates the full-text table from the chunks; ``search()`` parses the
query terms, runs the match, scores the rows, and returns ranked candidates.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from backend.app.retrieval.models import RetrievalCandidate


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "with", "who", "what", "which", "show", "me", "find", "about",
}

CODE_PATTERN = re.compile(r"\b[A-Za-z]{1,12}[-_/][A-Za-z0-9][A-Za-z0-9\-_/]*\b")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")
QUOTED_PATTERN = re.compile(r'"([^"]+)"')


@dataclass(frozen=True)
class KeywordIndexReport:
    """
    Report returned after rebuilding the keyword index.
    """

    metadata_db: str
    fts_table: str
    chunks_indexed: int
    status: str
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "metadata_db": self.metadata_db,
            "fts_table": self.fts_table,
            "chunks_indexed": self.chunks_indexed,
            "status": self.status,
            "error": self.error,
        }


class KeywordSearchIndex:
    """
    Local keyword search over the chunks table.

    The FTS table is rebuilt from SQLite chunk metadata. This keeps the source
    of truth simple: chunks remain in the normal chunks table; FTS is a derived
    search artifact.
    """

    def __init__(self, metadata_db_path: str | Path, table_name: str = "chunks_fts"):
        self.metadata_db_path = Path(metadata_db_path)
        self.table_name = table_name

    def rebuild(self) -> KeywordIndexReport:
        """
        Rebuild the FTS5 keyword index from the chunks table.
        """

        if not self.metadata_db_path.exists():
            return KeywordIndexReport(
                metadata_db=str(self.metadata_db_path),
                fts_table=self.table_name,
                chunks_indexed=0,
                status="error",
                error=f"Metadata DB not found: {self.metadata_db_path}",
            )

        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                self._create_table(conn)
                conn.execute(f"DELETE FROM {self.table_name};")

                rows = conn.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        source_system,
                        source_file,
                        record_type,
                        title,
                        citation_label,
                        text,
                        metadata_json
                    FROM chunks
                    ORDER BY rowid ASC;
                    """
                ).fetchall()

                conn.executemany(
                    f"""
                    INSERT INTO {self.table_name} (
                        chunk_id,
                        document_id,
                        source_system,
                        source_file,
                        record_type,
                        title,
                        citation_label,
                        text,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    rows,
                )
                conn.commit()

            return KeywordIndexReport(
                metadata_db=str(self.metadata_db_path),
                fts_table=self.table_name,
                chunks_indexed=len(rows),
                status="ok",
                error=None,
            )

        except sqlite3.OperationalError as exc:
            return KeywordIndexReport(
                metadata_db=str(self.metadata_db_path),
                fts_table=self.table_name,
                chunks_indexed=0,
                status="error",
                error=(
                    "Could not build SQLite FTS5 keyword index. "
                    "Your Python SQLite build may not include FTS5. "
                    f"Original error: {exc}"
                ),
            )

    def table_status(self) -> dict:
        """
        Return compact keyword-index status.
        """

        if not self.metadata_db_path.exists():
            return {"exists": False, "rows": 0, "error": "Metadata DB not found."}

        try:
            with sqlite3.connect(self.metadata_db_path) as conn:
                exists = conn.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table' AND name = ?;
                    """,
                    (self.table_name,),
                ).fetchone()

                if not exists:
                    return {"exists": False, "rows": 0, "error": None}

                rows = conn.execute(f"SELECT COUNT(*) FROM {self.table_name};").fetchone()[0]
                return {"exists": True, "rows": rows, "error": None}

        except Exception as exc:
            return {"exists": False, "rows": 0, "error": str(exc)}

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_system: str | None = None,
        record_type: str | None = None,
    ) -> list[RetrievalCandidate]:
        """
        Search keyword index.

        The method tries FTS5 first, then falls back to direct chunk scanning if
        the FTS table is missing or SQLite rejects the generated MATCH query.
        """

        query = query.strip()
        if not query:
            return []

        try:
            results = self._search_fts(
                query=query,
                top_k=top_k,
                source_system=source_system,
                record_type=record_type,
            )
            if results:
                return results
        except Exception:
            # Do not fail retrieval because FTS syntax is picky. The fallback is
            # slower but robust for this PoC dataset size.
            pass

        return self._search_fallback(
            query=query,
            top_k=top_k,
            source_system=source_system,
            record_type=record_type,
        )

    def _create_table(self, conn: sqlite3.Connection) -> None:
        """
        Create FTS5 table.
        """

        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}
            USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                source_system,
                source_file,
                record_type,
                title,
                citation_label UNINDEXED,
                text,
                metadata_json UNINDEXED
            );
            """
        )

    def _search_fts(
        self,
        query: str,
        top_k: int,
        source_system: str | None,
        record_type: str | None,
    ) -> list[RetrievalCandidate]:
        """
        Run FTS5 search.
        """

        match_query = self._build_match_query(query)
        if not match_query:
            return []

        where_clauses = [f"{self.table_name} MATCH ?"]
        params: list[object] = [match_query]

        if source_system:
            where_clauses.append("source_system = ?")
            params.append(source_system)

        if record_type:
            where_clauses.append("record_type = ?")
            params.append(record_type)

        params.append(max(top_k * 4, top_k))

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source_system,
                source_file,
                record_type,
                title,
                citation_label,
                text,
                metadata_json,
                bm25({self.table_name}) AS bm25_score
            FROM {self.table_name}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY bm25_score ASC
            LIMIT ?;
        """

        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        candidates: list[RetrievalCandidate] = []
        for row in rows:
            base_score = self._bm25_to_score(row["bm25_score"])
            refined_score, reasons = self._score_row(query=query, row=row, base_score=base_score)
            candidate = self._row_to_candidate(row)
            candidate.keyword_score = refined_score
            candidate.match_reasons.extend(["keyword_fts", *reasons])
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.keyword_score, reverse=True)
        return candidates[:top_k]

    def _search_fallback(
        self,
        query: str,
        top_k: int,
        source_system: str | None,
        record_type: str | None,
    ) -> list[RetrievalCandidate]:
        """
        Fallback keyword search by scanning chunks and scoring in Python.
        """

        where_clauses = []
        params: list[object] = []

        if source_system:
            where_clauses.append("source_system = ?")
            params.append(source_system)

        if record_type:
            where_clauses.append("record_type = ?")
            params.append(record_type)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT
                chunk_id,
                document_id,
                source_system,
                source_file,
                record_type,
                title,
                citation_label,
                text,
                metadata_json
            FROM chunks
            {where_sql};
        """

        with sqlite3.connect(self.metadata_db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()

        candidates: list[RetrievalCandidate] = []
        for row in rows:
            score, reasons = self._score_row(query=query, row=row, base_score=0.0)
            if score <= 0:
                continue

            candidate = self._row_to_candidate(row)
            candidate.keyword_score = score
            candidate.match_reasons.extend(["keyword_fallback", *reasons])
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.keyword_score, reverse=True)
        return candidates[:top_k]

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> RetrievalCandidate:
        """
        Convert SQLite row to RetrievalCandidate.
        """

        return RetrievalCandidate(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            source_system=str(row["source_system"] or ""),
            source_file=str(row["source_file"] or ""),
            record_type=str(row["record_type"] or ""),
            title=str(row["title"] or ""),
            citation_label=str(row["citation_label"] or ""),
            text=str(row["text"] or ""),
            metadata_json=str(row["metadata_json"] or ""),
        )

    @staticmethod
    def _bm25_to_score(value: float | int | None) -> float:
        """
        Convert SQLite bm25 value to a 0..1-ish score.

        FTS5 bm25 scores are lower-is-better and often negative. This conversion
        gives us a positive signal to combine with exact-match scoring.
        """

        if value is None:
            return 0.0

        magnitude = abs(float(value))
        return min(1.0, magnitude / (magnitude + 1.0))

    @staticmethod
    def _build_match_query(query: str) -> str:
        """
        Build a safe broad FTS5 query.

        We intentionally avoid raw punctuation-heavy terms because values like
        C-900 or HUNTER 2-S can break MATCH syntax. Exact matching for those is
        handled separately in Python scoring.
        """

        tokens = []
        for token in WORD_PATTERN.findall(query.lower()):
            if len(token) < 2 or token in STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)

        # Use OR to avoid missing results when the user enters a long query.
        return " OR ".join(f'"{token}"' for token in tokens[:16])

    @staticmethod
    def extract_terms(query: str) -> dict[str, list[str]]:
        """
        Extract useful keyword/exact terms from query.
        """

        quoted = [item.strip() for item in QUOTED_PATTERN.findall(query) if item.strip()]
        codes = [item.strip() for item in CODE_PATTERN.findall(query) if item.strip()]

        tokens = []
        for token in WORD_PATTERN.findall(query.lower()):
            if len(token) < 2 or token in STOPWORDS:
                continue
            if token not in tokens:
                tokens.append(token)

        exact_terms = []
        for term in [*quoted, *codes]:
            normalized = term.lower()
            if normalized not in [existing.lower() for existing in exact_terms]:
                exact_terms.append(term)

        return {
            "exact_terms": exact_terms,
            "tokens": tokens,
        }

    @classmethod
    def _score_row(cls, query: str, row: sqlite3.Row, base_score: float) -> tuple[float, list[str]]:
        """
        Score one row for keyword relevance.
        """

        terms = cls.extract_terms(query)
        exact_terms = terms["exact_terms"]
        tokens = terms["tokens"]

        title = str(row["title"] or "")
        source_file = str(row["source_file"] or "")
        text = str(row["text"] or "")

        title_l = title.lower()
        source_file_l = source_file.lower()
        text_l = text.lower()
        haystack_l = f"{title_l}\n{source_file_l}\n{text_l}"

        score = float(base_score)
        reasons: list[str] = []

        for term in exact_terms:
            term_l = term.lower()
            if term_l in title_l:
                score += 1.00
                reasons.append(f"exact_title:{term}")
            elif term_l in source_file_l:
                score += 0.85
                reasons.append(f"exact_file:{term}")
            elif term_l in text_l:
                score += 0.75
                reasons.append(f"exact_text:{term}")

        matched_tokens = 0
        for token in tokens:
            if token in title_l:
                matched_tokens += 1
                score += 0.18
            elif token in source_file_l:
                matched_tokens += 1
                score += 0.14
            elif token in text_l:
                matched_tokens += 1
                score += 0.08

        if tokens:
            coverage = matched_tokens / len(tokens)
            if coverage >= 0.75:
                score += 0.35
                reasons.append("high_keyword_coverage")
            elif coverage >= 0.40:
                score += 0.15
                reasons.append("partial_keyword_coverage")

        # Product/file-name style phrase: the full query is short enough and
        # appears as a substring.
        query_l = query.lower().strip()
        if 4 <= len(query_l) <= 120 and query_l in haystack_l:
            score += 0.75
            reasons.append("full_query_substring")

        return score, reasons
