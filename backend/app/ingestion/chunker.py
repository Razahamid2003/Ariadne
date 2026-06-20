"""Source-aware chunking.

Purpose
-------
Converts a loaded file into one document record and many well-formed chunks,
splitting text in a way that respects its structure instead of cutting blindly at
a fixed size.

What it does
------------
Keeps atomic records (spreadsheet rows, captions, OCR text) whole; splits prose at
headings and sentence boundaries with a small overlap between consecutive chunks so
context is not lost at the seams. Each chunk carries its section heading.

Flow
----
``build_document_and_chunks()`` builds the document record, then for each record
detects headings, packs sentences into size-bounded chunks with sentence-aligned
overlap, and labels each chunk's source system from its folder and file type.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.app.core.schemas import ChunkRecord, DocumentRecord
from backend.app.ingestion.loaders.base import LoadedFile
from backend.app.ingestion.metadata import chunk_id, document_id_from_path

# Records that represent a single logical unit and must not be sub-split
# (except as a last-resort safety split for pathological sizes).
ATOMIC_RECORD_TYPES = {
    "csv_row",
    "xlsx_row",
    "image_caption",
    "image_metadata",
    "image_ocr_text",
    "pdf_page_image_metadata",
    "pdf_page_vision_caption",
    "pdf_page_ocr_text",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S")
_LABEL_HEADING = re.compile(r"^[A-Z][A-Za-z0-9 /&()\-]{1,60}:\s*$")


# --------------------------------------------------------------------------- #
# Heading / section detection
# --------------------------------------------------------------------------- #
def _is_heading(line: str) -> bool:
    """Return True for structurally heading-like lines (domain-agnostic)."""

    value = line.strip()
    if not value or len(value) > 120:
        return False
    if _MARKDOWN_HEADING.match(value):
        return True
    if _NUMBERED_HEADING.match(value) and len(value) <= 90:
        return True
    if _LABEL_HEADING.match(value):
        return True
    # Short ALL-CAPS line with no terminal sentence punctuation.
    letters = [c for c in value if c.isalpha()]
    if letters and len(value) <= 80 and value == value.upper() and not value.endswith((".", "!", "?")):
        return True
    return False


def _clean_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line.strip()).rstrip(":").strip()


def _detect_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) sections using structural headings.

    Text before the first heading is returned with an empty heading.
    """

    lines = text.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_body: list[str] = []

    for line in lines:
        if _is_heading(line):
            if current_body or current_heading:
                sections.append((current_heading, current_body))
            current_heading = _clean_heading(line)
            current_body = []
        else:
            current_body.append(line)
    if current_body or current_heading:
        sections.append((current_heading, current_body))

    out: list[tuple[str, str]] = []
    for heading, body in sections:
        body_text = "\n".join(body).strip()
        if heading or body_text:
            out.append((heading, body_text))
    return out or [("", text.strip())]


# --------------------------------------------------------------------------- #
# Sentence-safe packing
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _hard_split(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Last-resort character split for a single sentence/token longer than the
    budget. Tries to break on whitespace near the boundary, never mid-word
    unless a single token itself exceeds max_chars."""

    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            space = text.rfind(" ", start + int(max_chars * 0.6), end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def _pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Pack sentences into <= max_chars chunks with sentence-aligned overlap."""

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> list[str]:
        nonlocal current, current_len
        if not current:
            return []
        chunks.append(" ".join(current).strip())
        # Build overlap tail from trailing sentences up to overlap_chars.
        tail: list[str] = []
        tail_len = 0
        for sentence in reversed(current):
            if tail_len + len(sentence) > overlap_chars and tail:
                break
            tail.insert(0, sentence)
            tail_len += len(sentence) + 1
        current = list(tail)
        current_len = sum(len(s) + 1 for s in current)
        return current

    for sentence in sentences:
        if len(sentence) > max_chars:
            # Emit what we have, then hard-split the oversize sentence.
            if current:
                flush()
                current, current_len = [], 0
            chunks.extend(_hard_split(sentence, max_chars, overlap_chars))
            continue
        if current_len + len(sentence) + 1 > max_chars and current:
            flush()
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# Per-record chunking
# --------------------------------------------------------------------------- #
def smart_chunk_record(
    text: str,
    record_type: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[tuple[str, str, str]]:
    """Chunk one record into (chunk_text, heading, strategy) tuples."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be >= 0 and < max_chars")

    body = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not body:
        return []

    rtype = (record_type or "").lower()

    # Atomic records: keep whole, safety-split only if pathologically large.
    if rtype in ATOMIC_RECORD_TYPES:
        if len(body) <= max_chars * 4:
            return [(body, "", "atomic")]
        return [(piece, "", "atomic_safety_split") for piece in _hard_split(body, max_chars, overlap_chars)]

    # Prose records: layout-aware section split, then sentence-safe packing.
    results: list[tuple[str, str, str]] = []
    for heading, section_body in _detect_sections(body):
        section_body = section_body.strip()
        if not section_body and not heading:
            continue
        full = f"{heading}\n{section_body}".strip() if heading else section_body
        if len(full) <= max_chars:
            results.append((full, heading, "section"))
            continue
        for piece in _pack_sentences(_split_sentences(section_body), max_chars, overlap_chars):
            chunk_text = f"{heading}\n{piece}".strip() if heading else piece
            results.append((chunk_text, heading, "section_split"))
    return results or [(body, "", "fallback")]


def split_text_into_chunks(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """Backward-compatible helper: sentence-safe chunking of free text.

    Retained for any external caller; internally the pipeline uses
    smart_chunk_record so chunk metadata (heading, strategy) is preserved.
    """

    return [c for c, _, _ in smart_chunk_record(text, "text_document", max_chars, overlap_chars)]


# --------------------------------------------------------------------------- #
# Source-system inference (folder-first, generic)
# --------------------------------------------------------------------------- #
def infer_source_system(path: Path, input_root: Path) -> str:
    """Infer a source-system label without any domain keyword maps.

    Priority:
        1. Top-level subfolder under input_root (any name, uppercased).
        2. File-extension family (IMAGES / DOCS / STRUCTURED).
        3. UNCLASSIFIED.
    """

    try:
        relative = path.relative_to(input_root)
    except ValueError:
        relative = Path(path.name)

    parts = [p for p in relative.parts[:-1] if p not in (".", "")]
    # Skip an internal extraction folder if present, use the next meaningful one.
    meaningful = [p for p in parts if not p.startswith("_")]
    if meaningful:
        return re.sub(r"[^A-Za-z0-9]+", "_", meaningful[0]).strip("_").upper() or "UNCLASSIFIED"

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
        return "IMAGES"
    if suffix in {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"}:
        return "DOCS"
    if suffix in {".csv", ".xlsx", ".xls", ".tsv"}:
        return "STRUCTURED"
    return "UNCLASSIFIED"


# --------------------------------------------------------------------------- #
# Document + chunk assembly
# --------------------------------------------------------------------------- #
def build_document_and_chunks(
    loaded_file: LoadedFile,
    input_root: str | Path,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    sensitivity: str = "internal",
) -> tuple[DocumentRecord, list[ChunkRecord]]:
    """Convert a LoadedFile into one DocumentRecord and many ChunkRecords."""

    source_path = loaded_file.source_path
    input_root_path = Path(input_root)
    document_id = document_id_from_path(source_path)
    source_system = infer_source_system(source_path, input_root_path)

    document = DocumentRecord(
        document_id=document_id,
        source_file=str(source_path),
        source_system=source_system,
        record_type="source_file",
        title=source_path.name,
        sensitivity=sensitivity,
        metadata={
            "file_name": source_path.name,
            "file_suffix": source_path.suffix.lower(),
            **loaded_file.document_metadata,
        },
    )

    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for record_index, record in enumerate(loaded_file.records):
        record_chunks = smart_chunk_record(
            text=record.text,
            record_type=record.record_type,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for record_chunk_index, (chunk_text, heading, strategy) in enumerate(record_chunks):
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id(document_id, chunk_index),
                    document_id=document_id,
                    text=chunk_text,
                    source_file=str(source_path),
                    source_system=source_system,
                    record_type=record.record_type,
                    title=record.title or source_path.name,
                    chunk_index=chunk_index,
                    sensitivity=sensitivity,
                    metadata={
                        "file_name": source_path.name,
                        "file_suffix": source_path.suffix.lower(),
                        "record_index": record_index,
                        "record_chunk_index": record_chunk_index,
                        "chunk_strategy": strategy,
                        "section_heading": heading,
                        **record.metadata,
                    },
                )
            )
            chunk_index += 1

    return document, chunks
