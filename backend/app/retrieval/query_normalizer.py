"""Query normalization.

Purpose
-------
Expands common abbreviations and informal terms in a query before search, so
everyday phrasing still matches the documents.

What it does
------------
Appends configured expansions uniformly to every query. It is transparent and
config-driven, and does not guess intent or inject hidden hints.
"""

from __future__ import annotations

import re


def normalize_query(query: str, expansions: dict[str, str] | None) -> str:
    """Return query with any configured expansions appended.

    Expansions are additive (the original terms are kept). Case-insensitive
    whole-word match. If no expansions configured, returns query unchanged.
    """

    if not expansions or not query:
        return query

    q = query.strip()
    additions: list[str] = []

    for term, expansion in expansions.items():
        if not term or not expansion:
            continue
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b", re.IGNORECASE)
        if pattern.search(q):
            # Only add if the expansion words aren't already in the query
            for word in expansion.split():
                if not re.search(r"\b" + re.escape(word) + r"\b", q, re.IGNORECASE):
                    additions.append(word)

    if not additions:
        return q

    return q + " " + " ".join(additions)
