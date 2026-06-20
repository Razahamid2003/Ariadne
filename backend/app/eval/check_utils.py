"""Shared evaluation helper functions.

Purpose
-------
Common text and scoring helpers used by the answer- and retrieval-level checks.

What it does
------------
Provides term matching, confidence comparison, latency percentiles, source-name
extraction, and detectors for bad output patterns (mixed answer/no-evidence text,
raw chunk-slice artifacts).
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any, Iterable

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def text_blob(value: Any) -> str:
    """Return a flattened text view over nested JSON-like values."""

    parts: list[str] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            parts.append(item)
            return
        if isinstance(item, (int, float, bool)):
            parts.append(str(item))
            return
        if isinstance(item, dict):
            for nested in item.values():
                walk(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                walk(nested)
            return
        parts.append(str(item))

    walk(value)
    return "\n".join(parts)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def contains_term(blob: str, term: str) -> bool:
    return norm(term) in norm(blob)


def contains_any(blob: str, terms: Iterable[str]) -> bool:
    return any(contains_term(blob, term) for term in terms)


def missing_all_terms(blob: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if not contains_term(blob, term)]


def present_terms(blob: str, terms: Iterable[str]) -> list[str]:
    return [term for term in terms if contains_term(blob, term)]


def min_confidence_ok(actual: str | None, minimum: str | None) -> bool:
    if not minimum:
        return True
    return CONFIDENCE_RANK.get(norm(actual), -1) >= CONFIDENCE_RANK.get(norm(minimum), 99)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    return {
        "min": min(values),
        "p50": float(median(values)),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def extract_source_names_from_chat(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for doc in response.get("source_documents") or []:
        if isinstance(doc, dict):
            for key in ("display_name", "title", "source_file", "file_name"):
                value = doc.get(key)
                if value:
                    names.append(str(value))
                    break
    for item in response.get("evidence") or []:
        if isinstance(item, dict):
            value = item.get("source_file") or item.get("title") or item.get("document_id")
            if value:
                names.append(str(value))
    return list(dict.fromkeys(names))


def extract_source_names_from_search(response: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for result in response.get("results") or []:
        if isinstance(result, dict):
            value = result.get("source_file") or result.get("title") or result.get("document_id")
            if value:
                names.append(str(value))
    return list(dict.fromkeys(names))


def looks_like_mixed_no_answer(answer: str, status: str | None = None) -> bool:
    """Detect the bad pattern: factual answer text plus a no-evidence conclusion."""

    text = norm(answer)
    if not text:
        return False
    no_answer_markers = (
        "not enough evidence",
        "does not contain enough information",
        "cannot answer this reliably",
        "no indexed evidence",
    )
    has_no_answer_marker = any(marker in text for marker in no_answer_markers) or norm(status) == "no_answer"
    if not has_no_answer_marker:
        return False

    # Factual-looking content before/after the no-answer marker. This is a heuristic,
    # but it catches the exact regression where an answer listed dates/courses and
    # then said not enough evidence.
    factual_markers = (
        "duration:",
        "degree:",
        "experience:",
        "education:",
        "employee id:",
        "certification:",
        "source 1",
        "september",
        "january",
        "february",
        "march",
        "april",
        "may ",
        "june",
        "july",
        "august",
        "october",
        "november",
        "december",
    )
    return any(marker in text for marker in factual_markers)


def looks_like_raw_excerpt_artifact(answer: str) -> bool:
    """Detect raw chunk-slice artifacts that should not appear as final answers."""

    text = str(answer or "")
    if not text.strip():
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    artifact_patterns = [
        r"^[a-z]\s+by\s+",  # e.g. "t by emotional tone..."
        r"^[a-z]{1,2}\s+(?:with|from|and|or|to|by|of)\b",
        r"^\.\.\.",
        r"^[,;:)\]]",
    ]
    first_content = lines[0]
    if any(re.search(pattern, first_content) for pattern in artifact_patterns):
        return True

    # Repeated line endings with ellipses are usually raw excerpt dumps. A single
    # ellipsis in a polished answer is tolerated.
    ellipsis_endings = sum(1 for line in lines if line.endswith("...") or line.endswith("…"))
    return ellipsis_endings >= 2
