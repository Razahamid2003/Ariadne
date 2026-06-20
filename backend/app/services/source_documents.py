"""Source-document aggregation.

Purpose
-------
Groups the chunk-level evidence behind an answer into clean, document-level source
cards for the UI, while the model keeps citing precise chunks internally.

What it does
------------
Builds user-facing source cards with friendly document names and previews, and
maps each chunk citation to its document so the UI can show "where this came from."

Flow
----
Cited evidence is grouped by source document; each group becomes a card the UI
shows under the answer, hiding internal chunk IDs from the reader.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


def _get(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _text_preview(item: Any, preview_chars: int) -> str:
    text = _get(item, "text", "") or _get(item, "text_preview", "") or ""
    return str(text)[:preview_chars]


def _display_name_for_source(source_file: str, title: str, document_id: str) -> str:
    """Return a user-facing document name, not a full local path."""

    if source_file:
        normalized = source_file.replace("\\", "/").rstrip("/")
        name = Path(normalized).name
        if name:
            return name
    return title or document_id or "Local source"


def build_source_documents(evidence: Iterable[Any], preview_chars: int = 900) -> list[dict[str, Any]]:
    """Group evidence chunks into user-facing source-document cards."""

    grouped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    for item in evidence or []:
        document_id = str(_get(item, "document_id", "") or "")
        source_file = str(_get(item, "source_file", "") or "")
        title = str(_get(item, "title", "") or "")
        key = document_id or source_file or title or str(_get(item, "citation_label", "local-source"))

        if key not in grouped:
            display_name = _display_name_for_source(source_file, title, document_id)
            grouped[key] = {
                "source_index": len(grouped) + 1,
                "display_label": f"Source {len(grouped) + 1}",
                "display_name": display_name,
                "document_id": document_id,
                "source_file": source_file,
                "full_path": source_file,
                "title": title,
                "source_systems": [],
                "record_types": [],
                "citation_labels": [],
                "chunk_count": 0,
                "previews": [],
                "preview": "",
            }

        doc = grouped[key]
        doc["chunk_count"] += 1

        source_system = str(_get(item, "source_system", "") or "")
        record_type = str(_get(item, "record_type", "") or "")
        citation = str(_get(item, "citation_label", "") or "")
        preview = _text_preview(item, preview_chars)

        if source_system and source_system not in doc["source_systems"]:
            doc["source_systems"].append(source_system)
        if record_type and record_type not in doc["record_types"]:
            doc["record_types"].append(record_type)
        if citation and citation not in doc["citation_labels"]:
            doc["citation_labels"].append(citation)
        if preview and preview not in doc["previews"] and len(doc["previews"]) < 3:
            doc["previews"].append(preview)

    for doc in grouped.values():
        doc["preview"] = "\n\n".join(doc.pop("previews", []))[:preview_chars]

    return list(grouped.values())


def citation_to_source_map(source_documents: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return a chunk-citation -> document source lookup."""

    mapping: dict[str, dict[str, Any]] = {}
    for doc in source_documents or []:
        for label in doc.get("citation_labels", []) or []:
            mapping[str(label)] = {
                "display_label": doc.get("display_label") or "Source",
                "display_name": doc.get("display_name") or "Local source",
                "source_index": doc.get("source_index"),
                "document_id": doc.get("document_id") or doc.get("source_file") or "",
            }
    return mapping
