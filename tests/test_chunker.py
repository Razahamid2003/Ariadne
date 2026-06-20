"""Tests for the source-aware chunker.

Purpose
-------
Checks that atomic records stay whole, oversize records split safely, prose splits
at headings, consecutive chunks overlap, short documents stay in one chunk, and
source-system inference stays generic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ingestion.chunker import (
    build_document_and_chunks,
    infer_source_system,
    smart_chunk_record,
)
from backend.app.ingestion.loaders.base import LoadedFile, LoadedRecord


def test_atomic_row_kept_whole():
    text = "Employee ID: EMP-102\nName: A. Khan\nCertification: Safety Eng\nDepartment: Automation\nAvailable From: 2026-07-01"
    chunks = smart_chunk_record(text, "csv_row", max_chars=80, overlap_chars=10)
    assert len(chunks) == 1, f"a CSV row must stay one chunk, got {len(chunks)}"
    assert chunks[0][2] == "atomic"
    assert chunks[0][0] == text.strip()
    print(f"  csv_row kept whole as 1 atomic chunk (len {len(text)} > max 80)  ✓")


def test_oversize_atomic_safety_split():
    big = "col: " + ("x " * 1000)  # ~2005 chars, > 4*max
    chunks = smart_chunk_record(big, "xlsx_row", max_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(c[2] == "atomic_safety_split" for c in chunks)
    print(f"  pathologically large row safety-split into {len(chunks)} pieces  ✓")


def test_heading_aware_prose_split():
    text = (
        "# Overview\n"
        "This system validates retrieval. It runs locally. It avoids cloud calls. "
        "It supports many formats. It is configurable. It is auditable.\n\n"
        "# Procedure\n"
        "First ingest the files. Then build the indexes. Then run a query. "
        "Then inspect the citations. Then tune the thresholds. Then repeat as needed."
    )
    chunks = smart_chunk_record(text, "pdf_page", max_chars=120, overlap_chars=30)
    assert len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}"
    # Every chunk from a headed section must carry its heading.
    headings = {c[1] for c in chunks}
    assert "Overview" in headings and "Procedure" in headings, headings
    for chunk_text, heading, strategy in chunks:
        if heading:
            assert chunk_text.startswith(heading), f"chunk should carry heading: {chunk_text[:40]!r}"
    # Sentence-safe: no split chunk should end mid-word (ends with . or is whole).
    for chunk_text, heading, strategy in chunks:
        body = chunk_text[len(heading):].strip() if heading else chunk_text
        if strategy == "section_split":
            assert body.endswith((".", "!", "?")), f"not sentence-safe: ...{body[-30:]!r}"
    print(f"  prose split into {len(chunks)} heading-carrying, sentence-safe chunks  ✓")


def test_overlap_between_consecutive_chunks():
    sentences = " ".join(f"Sentence number {i} provides distinct evidence content here." for i in range(1, 13))
    chunks = smart_chunk_record(sentences, "text_document", max_chars=160, overlap_chars=70)
    assert len(chunks) >= 2
    # Consecutive chunks should share at least one sentence (overlap).
    overlaps = 0
    for a, b in zip(chunks, chunks[1:]):
        a_sents = set(s.strip() for s in a[0].split(".") if s.strip())
        b_sents = set(s.strip() for s in b[0].split(".") if s.strip())
        if a_sents & b_sents:
            overlaps += 1
    assert overlaps >= 1, "expected sentence-aligned overlap between consecutive chunks"
    print(f"  sentence-aligned overlap present across {overlaps} chunk boundaries  ✓")


def test_short_doc_single_chunk():
    chunks = smart_chunk_record("A short note that easily fits.", "text_document", max_chars=200, overlap_chars=20)
    assert len(chunks) == 1 and chunks[0][2] == "section"
    print("  short document -> single chunk  ✓")


def test_source_system_inference_is_generic():
    root = Path("/data/input")
    # folder-based: any subfolder name becomes the source system
    assert infer_source_system(Path("/data/input/Finance/q1.pdf"), root) == "FINANCE"
    assert infer_source_system(Path("/data/input/HR Records/list.csv"), root) == "HR_RECORDS"
    # underscore extraction folder skipped, next meaningful folder used
    assert infer_source_system(Path("/data/input/_extracted/Reports/r.pdf"), root) == "REPORTS"
    # flat files -> extension family, NOT content-based guessing
    assert infer_source_system(Path("/data/input/asset_machine_repair.csv"), root) == "STRUCTURED"
    assert infer_source_system(Path("/data/input/manual.pdf"), root) == "DOCS"
    assert infer_source_system(Path("/data/input/photo.jpg"), root) == "IMAGES"
    print("  source-system inference is folder/extension based, no domain keywords  ✓")


def test_build_document_and_chunks_end_to_end():
    lf = LoadedFile(
        source_path=Path("/data/input/Policies/handbook.txt"),
        records=[
            LoadedRecord(text="# Intro\nThis is the intro. It has two sentences.", record_type="text_document", title="handbook"),
            LoadedRecord(text="Name: X\nRole: Y\nDept: Z", record_type="csv_row", title="row 1"),
        ],
        document_metadata={"pages": 1},
    )
    doc, chunks = build_document_and_chunks(lf, "/data/input", max_chars=200, overlap_chars=20)
    assert doc.source_system == "POLICIES"
    assert len(chunks) == 2
    assert chunks[0].metadata["section_heading"] == "Intro"
    assert chunks[1].metadata["chunk_strategy"] == "atomic"
    # chunk_index is monotonic and citation labels are stable
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert chunks[0].citation_label().startswith("[POLICIES: ")
    print(f"  end-to-end: doc.source_system={doc.source_system}, {len(chunks)} chunks, labels stable  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} chunker tests (no model runtime)...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
