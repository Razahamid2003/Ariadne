"""Ingestion pipeline.

Purpose
-------
The end-to-end routine that turns a folder of source files into stored documents
and chunks ready for indexing.

What it does
------------
Walks the input directory, loads each supported file, chunks it, classifies its
type, writes inspection outputs, and stores everything in the metadata database,
returning a summary report.

Flow
----
``run_ingestion()`` iterates candidate files, calls the right loader, builds the
document and chunks, classifies the document type, merges that type onto the
chunks, and persists the results while recording counts and any errors.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from backend.app.core.schemas import ChunkRecord, DocumentRecord
from backend.app.ingestion.chunker import build_document_and_chunks
from backend.app.ingestion.registry import LoaderRegistry
from backend.app.persistence.sqlite_metadata_store import SQLiteMetadataStore


@dataclass(frozen=True)
class IngestionReport:
    """
    Summary of one ingestion run.
    """

    input_dir: str
    output_dir: str
    metadata_db: str
    supported_extensions: list[str]
    files_seen: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    documents_created: int = 0
    chunks_created: int = 0
    sqlite_documents_total: int = 0
    sqlite_chunks_total: int = 0
    sqlite_ingestion_runs_total: int = 0
    ingestion_run_id: str | None = None
    errors: list[str] = field(default_factory=list)


def _classify_document_metadata(chunks, auto_metadata_config, llm_client) -> dict:
    """Classify one document's metadata (e.g. document_type) via the LLM.

    Returns a {field: value} dict to merge into document/chunk metadata, or {}
    when disabled or unavailable. Always graceful: never raises.
    """

    if not auto_metadata_config or not getattr(auto_metadata_config, "enabled", False):
        return {}
    if llm_client is None:
        return {}
    try:
        import asyncio

        from backend.app.ingestion.document_classifier import MetadataField, classify_document

        fields = [MetadataField.from_config(f) for f in getattr(auto_metadata_config, "fields", []) or []]
        if not fields:
            return {}
        sample_chars = int(getattr(auto_metadata_config, "sample_chars", 4000))
        sample = ""
        for chunk in chunks:
            if len(sample) >= sample_chars:
                break
            sample += (chunk.text or "") + "\n"
        sample = sample[:sample_chars]

        return asyncio.run(classify_document(sample, fields, llm_client, enabled=True))
    except Exception:
        return {}


def iter_candidate_files(input_dir: str | Path) -> Iterable[Path]:
    """
    Yield candidate files from input_dir recursively.

    Hidden files and internal runtime files are skipped.

    Skipped examples:
        - .gitkeep
        - .DS_Store
        - files inside __pycache__
    """

    root = Path(input_dir)

    if not root.exists():
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if "__pycache__" in path.parts:
            continue

        yield path


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """
    Write rows to a JSONL file.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict) -> None:
    """
    Write a JSON file.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def run_ingestion(
    input_dir: str | Path,
    output_dir: str | Path,
    metadata_db_path: str | Path,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    vision_config=None,
    ocr_config=None,
    auto_metadata_config=None,
    llm_client=None,
) -> IngestionReport:
    """
    Run unified ingestion for all currently supported file types.

    Args:
        input_dir:
            Directory containing source files.

        output_dir:
            Directory where JSONL inspection files are written.

        metadata_db_path:
            SQLite database path.

        max_chars:
            Target maximum chunk size.

        overlap_chars:
            Chunk overlap.

    Returns:
        IngestionReport:
            Summary of processed files and SQLite totals.
    """

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    registry = LoaderRegistry(
        vision_config=vision_config,
        ocr_config=ocr_config,
    )

    supported_extensions = sorted(registry.supported_extensions())

    documents: list[DocumentRecord] = []
    chunks: list[ChunkRecord] = []
    errors: list[str] = []

    files_seen = 0
    files_processed = 0
    files_skipped = 0

    for file_path in iter_candidate_files(input_root):
        loader = registry.get_loader(file_path)

        if loader is None:
            continue

        files_seen += 1

        try:
            loaded_file = loader.load(file_path)

            document, document_chunks = build_document_and_chunks(
                loaded_file=loaded_file,
                input_root=input_root,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                sensitivity="internal",
            )

            if not document_chunks:
                files_skipped += 1
                errors.append(f"No chunks created for: {file_path}")
                continue

            # Auto-metadata: classify the document's type once,
            # store it on the document and propagate to every chunk so it can be
            # used as a soft retrieval signal and as generation context. Graceful:
            # any failure falls back to the configured default and never blocks.
            extracted_metadata = _classify_document_metadata(
                document_chunks, auto_metadata_config, llm_client
            )
            if extracted_metadata:
                document.metadata.update(extracted_metadata)
                for chunk in document_chunks:
                    chunk.metadata.update(extracted_metadata)

            documents.append(document)
            chunks.extend(document_chunks)
            files_processed += 1

        except Exception as exc:
            files_skipped += 1
            errors.append(f"{file_path}: {exc}")

    output_root.mkdir(parents=True, exist_ok=True)

    documents_path = output_root / "documents.jsonl"
    chunks_path = output_root / "chunks.jsonl"
    report_path = output_root / "ingestion_report.json"

    write_jsonl(
        documents_path,
        [document.model_dump(mode="json") for document in documents],
    )

    write_jsonl(
        chunks_path,
        [chunk.model_dump(mode="json") for chunk in chunks],
    )

    store = SQLiteMetadataStore(metadata_db_path)
    store.initialize()
    store.upsert_documents(documents)
    store.upsert_chunks(chunks)

    initial_report = IngestionReport(
        input_dir=str(input_root),
        output_dir=str(output_root),
        metadata_db=str(metadata_db_path),
        supported_extensions=supported_extensions,
        files_seen=files_seen,
        files_processed=files_processed,
        files_skipped=files_skipped,
        documents_created=len(documents),
        chunks_created=len(chunks),
        sqlite_documents_total=store.count_documents(),
        sqlite_chunks_total=store.count_chunks(),
        sqlite_ingestion_runs_total=store.count_ingestion_runs(),
        ingestion_run_id=None,
        errors=errors,
    )

    run_id = store.save_ingestion_run(initial_report)

    final_report = IngestionReport(
        input_dir=initial_report.input_dir,
        output_dir=initial_report.output_dir,
        metadata_db=initial_report.metadata_db,
        supported_extensions=initial_report.supported_extensions,
        files_seen=initial_report.files_seen,
        files_processed=initial_report.files_processed,
        files_skipped=initial_report.files_skipped,
        documents_created=initial_report.documents_created,
        chunks_created=initial_report.chunks_created,
        sqlite_documents_total=store.count_documents(),
        sqlite_chunks_total=store.count_chunks(),
        sqlite_ingestion_runs_total=store.count_ingestion_runs(),
        ingestion_run_id=run_id,
        errors=initial_report.errors,
    )

    write_json(report_path, final_report.__dict__)

    return final_report