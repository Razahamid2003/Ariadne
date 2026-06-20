"""Management command-line interface.

Purpose
-------
A single command entry point for everyday operations: inventory, extract, ingest,
build indexes, search, generate answers, inspect, clean, rebuild, and status.

What it does
------------
Each subcommand maps to one operation. ``rebuild`` chains clean, ingest, keyword
index, and vector index so one command produces a ready-to-use system.

Usage
-----
    python scripts/manage_rag.py <command> --config config/client.yaml
    python scripts/manage_rag.py rebuild --fresh
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.embeddings.local_embedding_model import SentenceTransformersEmbeddingModel
from backend.app.indexing.vector_index import LocalVectorIndex
from backend.app.indexing.index_status import get_index_lifecycle_status
from backend.app.ingestion.pipeline import run_ingestion
from backend.app.intake.file_registry import FileRegistry
from backend.app.intake.incremental_ingestion import IncrementalIngestionManager
from backend.app.ingestion.registry import LoaderRegistry
from backend.app.intake.archive_extractor import extract_archives
from backend.app.ocr.local_ocr import LocalOcrClient
from backend.app.persistence.sqlite_metadata_store import SQLiteMetadataStore
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.keyword_index import KeywordSearchIndex
from backend.app.retrieval.models import HybridSearchRequest
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.models import RAGAnswerRequest
from backend.app.vision.local_vision_client import LocalVisionClient


ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown"}
STRUCTURED_EXTENSIONS = {".csv", ".xlsx"}


def print_json(data: Any) -> None:
    """Print JSON consistently."""

    print(json.dumps(data, indent=2, ensure_ascii=False))


def load_config(config_path: str):
    """Load settings from YAML."""

    return load_settings(config_path)


def table_exists_for_cli(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?;",
        (table_name,),
    ).fetchone()
    return row is not None


def safe_count_table(conn: sqlite3.Connection, table_name: str) -> int:
    if not table_exists_for_cli(conn, table_name):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0])


def is_runtime_or_hidden_file(path: Path) -> bool:
    """Skip files that should not be treated as input content."""

    if path.name.startswith("."):
        return True

    if "__pycache__" in path.parts:
        return True

    return False


def categorize_extension(extension: str) -> str:
    """Categorize file extension for inventory output."""

    if extension in ARCHIVE_EXTENSIONS:
        return "archive"

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in DOCUMENT_EXTENSIONS:
        return "document"

    if extension in STRUCTURED_EXTENSIONS:
        return "structured"

    if extension == "":
        return "no_extension"

    return "other"


def command_inventory(args: argparse.Namespace) -> int:
    """Inventory input files."""

    settings = load_config(args.config)
    input_root = Path(settings.paths.input_data)

    registry = LoaderRegistry(
        vision_config=settings.vision,
        ocr_config=settings.ocr,
    )
    supported_extensions = registry.supported_extensions()

    files = [
        path
        for path in sorted(input_root.rglob("*"))
        if path.is_file() and not is_runtime_or_hidden_file(path)
    ]

    extension_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    supported_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    samples_by_extension: dict[str, list[str]] = defaultdict(list)

    for path in files:
        extension = path.suffix.lower()
        extension_key = extension or "[no extension]"
        category = categorize_extension(extension)

        extension_counts[extension_key] += 1
        category_counts[category] += 1

        if extension in supported_extensions:
            supported_counts[extension] += 1
        else:
            unsupported_counts[extension_key] += 1

        if len(samples_by_extension[extension_key]) < args.sample_limit:
            samples_by_extension[extension_key].append(str(path))

    output = {
        "input_dir": str(input_root),
        "total_files": len(files),
        "supported_extensions": sorted(supported_extensions),
        "counts_by_category": dict(category_counts),
        "counts_by_extension": dict(sorted(extension_counts.items())),
        "supported_counts": dict(sorted(supported_counts.items())),
        "unsupported_counts": dict(sorted(unsupported_counts.items())),
        "samples_by_extension": dict(samples_by_extension),
    }

    print_json(output)

    if unsupported_counts:
        print("[WARN] Unsupported file types are present.")
    else:
        print("[PASS] All discovered file extensions are supported.")

    return 0


def command_extract(args: argparse.Namespace) -> int:
    """Extract archives."""

    settings = load_config(args.config)

    if not settings.archives.enabled:
        print("[INFO] Archive extraction is disabled in config.")
        return 0

    report = extract_archives(
        input_dir=settings.paths.input_data,
        extract_dir=settings.archives.extract_dir,
        clear_extract_dir=args.clear,
    )

    print_json(report.__dict__)

    if report.errors:
        print("[WARN] Archive extraction completed with warnings.")
    else:
        print("[PASS] Archive extraction completed successfully.")

    return 0


def command_plan_ingest(args: argparse.Namespace) -> int:
    """Show which files would be ingested, skipped, or removed."""

    settings = load_config(args.config)
    manager = IncrementalIngestionManager(
        input_dir=settings.paths.input_data,
        output_dir=settings.paths.processed_data,
        metadata_db_path=settings.paths.metadata_db,
        max_chars=settings.ingestion.max_chars,
        overlap_chars=settings.ingestion.overlap_chars,
        vision_config=settings.vision,
        ocr_config=settings.ocr,
        work_dir=settings.file_tracking.work_dir,
    )

    plan = manager.plan()
    print_json(plan.to_dict())

    if plan.files_new or plan.files_changed or plan.files_missing:
        print("[INFO] Changes detected. Incremental ingest is needed.")
    else:
        print("[PASS] No file changes detected.")

    return 0


def _build_ingestion_llm_client(settings):
    """LLM client for auto-metadata classification, or None if disabled."""
    auto = getattr(settings.ingestion, "auto_metadata", None)
    if not auto or not getattr(auto, "enabled", False):
        return None
    try:
        from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient
        return OpenAICompatibleLLMClient(settings.llm)
    except Exception:
        return None


def command_ingest(args: argparse.Namespace) -> int:
    """Run ingestion.

    Default behavior is incremental ingestion: only new or
    changed files are processed, and missing/deleted file rows are removed from
    SQLite. Use --full-scan to call the legacy all-files ingestion pipeline.
    Use --force to reprocess all supported files through the incremental manager.
    """

    settings = load_config(args.config)
    full_scan = bool(getattr(args, "full_scan", False))
    force = bool(getattr(args, "force", False))

    if settings.file_tracking.enabled and not full_scan:
        manager = IncrementalIngestionManager(
            input_dir=settings.paths.input_data,
            output_dir=settings.paths.processed_data,
            metadata_db_path=settings.paths.metadata_db,
            max_chars=settings.ingestion.max_chars,
            overlap_chars=settings.ingestion.overlap_chars,
            vision_config=settings.vision,
            ocr_config=settings.ocr,
            work_dir=settings.file_tracking.work_dir,
            auto_metadata_config=getattr(settings.ingestion, "auto_metadata", None),
            llm_client=_build_ingestion_llm_client(settings),
        )
        report = manager.ingest(force=force)
        print_json(report.to_dict())

        if report.errors:
            print("[WARN] Incremental ingestion completed with warnings/errors.")
            return 1

        if report.index_status == "stale":
            print("[INFO] Search indexes are stale. Run build-keyword-index and build-index.")

        print("[PASS] Incremental ingestion completed successfully.")
        print(f"[INFO] Metadata DB: {settings.paths.metadata_db}")
        return 0

    report = run_ingestion(
        input_dir=settings.paths.input_data,
        output_dir=settings.paths.processed_data,
        metadata_db_path=settings.paths.metadata_db,
        max_chars=settings.ingestion.max_chars,
        overlap_chars=settings.ingestion.overlap_chars,
        vision_config=settings.vision,
        ocr_config=settings.ocr,
    )

    print_json(report.__dict__)

    if report.errors:
        print("[WARN] Full-scan ingestion completed with warnings/errors.")
        return 1

    print("[PASS] Full-scan ingestion completed successfully.")
    print(f"[INFO] Metadata DB: {settings.paths.metadata_db}")
    return 0


def command_count(args: argparse.Namespace) -> int:
    """Count chunk record types."""

    settings = load_config(args.config)
    db_path = Path(settings.paths.metadata_db)

    if not db_path.exists():
        print(f"[FAIL] Metadata DB not found: {db_path}")
        return 1

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT record_type, COUNT(*) AS count
            FROM chunks
            GROUP BY record_type
            ORDER BY count DESC;
            """
        ).fetchall()

    output = [{"record_type": row[0], "count": row[1]} for row in rows]
    print_json(output)
    return 0


def command_inspect_metadata(args: argparse.Namespace) -> int:
    """Inspect recent chunks in metadata DB."""

    settings = load_config(args.config)
    db_path = Path(settings.paths.metadata_db)

    if not db_path.exists():
        print(f"[FAIL] Metadata DB not found: {db_path}")
        return 1

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        documents = conn.execute("SELECT COUNT(*) FROM documents;").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
        ingestion_runs = conn.execute("SELECT COUNT(*) FROM ingestion_runs;").fetchone()[0]

        rows = conn.execute(
            """
            SELECT
                chunk_id,
                document_id,
                source_system,
                source_file,
                record_type,
                title,
                chunk_index,
                citation_label,
                substr(text, 1, ?) AS text_preview
            FROM chunks
            ORDER BY rowid DESC
            LIMIT ?;
            """,
            (args.preview_chars, args.limit),
        ).fetchall()

    output = {
        "metadata_db": str(db_path),
        "documents": documents,
        "chunks": chunks,
        "ingestion_runs": ingestion_runs,
        "recent_chunks": [dict(row) for row in rows],
    }

    print_json(output)
    return 0


def command_build_keyword_index(args: argparse.Namespace) -> int:
    """Build SQLite FTS5 keyword index."""

    settings = load_config(args.config)
    index = KeywordSearchIndex(
        metadata_db_path=settings.paths.metadata_db,
        table_name=settings.retrieval.keyword_table,
    )

    report = index.rebuild()
    print_json(report.to_dict())

    if report.status != "ok":
        print("[FAIL] Keyword index build failed.")
        return 1

    print("[PASS] Keyword index built successfully.")
    return 0


def command_build_index(args: argparse.Namespace) -> int:
    """Build local vector index."""

    settings = load_config(args.config)

    store = SQLiteMetadataStore(settings.paths.metadata_db)
    chunks = store.list_chunks_for_indexing()

    if not chunks:
        print("[FAIL] No chunks found. Run ingestion first.")
        return 1

    texts = [chunk["text"] for chunk in chunks]

    print(f"[INFO] Chunks to embed: {len(texts)}")
    print(f"[INFO] Embedding provider: {settings.embeddings.provider}")
    print(f"[INFO] Embedding model: {settings.embeddings.model_name_or_path}")
    print(f"[INFO] Device: {settings.embeddings.device}")
    print(f"[INFO] Batch size: {settings.embeddings.batch_size}")

    embedding_model = SentenceTransformersEmbeddingModel(settings.embeddings)
    batch = embedding_model.encode(texts)

    metadata_rows = [
        {
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_system": chunk["source_system"],
            "source_file": chunk["source_file"],
            "record_type": chunk["record_type"],
            "title": chunk["title"],
            "chunk_index": chunk["chunk_index"],
            "sensitivity": chunk["sensitivity"],
            "citation_label": chunk["citation_label"],
            "text": chunk["text"],
            "metadata_json": chunk["metadata_json"],
        }
        for chunk in chunks
    ]

    index = LocalVectorIndex(
        index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
    )

    index.save(batch.vectors, metadata_rows)

    output = {
        "chunks_indexed": batch.count,
        "embedding_dimension": batch.dimension,
        "index_dir": settings.vector_index.index_dir,
        "embeddings_path": str(index.embeddings_path),
        "metadata_path": str(index.metadata_path),
    }

    print_json(output)
    print("[PASS] Vector index built successfully.")
    return 0


def command_inspect_index(args: argparse.Namespace) -> int:
    """Inspect local vector index."""

    settings = load_config(args.config)

    index_dir = Path(settings.vector_index.index_dir)
    embeddings_path = index_dir / settings.vector_index.embeddings_file
    metadata_path = index_dir / settings.vector_index.metadata_file

    if not embeddings_path.exists():
        print(f"[FAIL] Missing embeddings file: {embeddings_path}")
        return 1

    if not metadata_path.exists():
        print(f"[FAIL] Missing metadata file: {metadata_path}")
        return 1

    vectors = np.load(embeddings_path)

    metadata_rows = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                metadata_rows.append(json.loads(line))

    source_counts = Counter(row.get("source_system") for row in metadata_rows)
    record_type_counts = Counter(row.get("record_type") for row in metadata_rows)

    output = {
        "index_dir": str(index_dir),
        "embeddings_path": str(embeddings_path),
        "metadata_path": str(metadata_path),
        "vector_shape": list(vectors.shape),
        "metadata_rows": len(metadata_rows),
        "source_counts": dict(source_counts),
        "record_type_counts": dict(record_type_counts),
    }

    print_json(output)

    if vectors.shape[0] != len(metadata_rows):
        print("[FAIL] Vector count does not match metadata rows.")
        return 1

    print("[PASS] Vector index looks valid.")
    return 0


def command_search(args: argparse.Namespace) -> int:
    """Search local vector index only."""

    settings = load_config(args.config)

    print(f"[INFO] Query: {args.query}")
    print(f"[INFO] Top K: {args.top_k}")
    print(f"[INFO] Source filter: {args.source_system}")
    print(f"[INFO] Record type filter: {args.record_type}")
    print(f"[INFO] Embedding model: {settings.embeddings.model_name_or_path}")

    embedding_model = SentenceTransformersEmbeddingModel(settings.embeddings)
    query_batch = embedding_model.encode([args.query], show_progress_bar=False)

    index = LocalVectorIndex(
        index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
    )

    results = index.search(
        query_vector=query_batch.vectors,
        top_k=args.top_k,
        source_system=args.source_system,
        record_type=args.record_type,
    )

    output = [
        {
            "rank": rank,
            "score": round(result.score, 4),
            "chunk_id": result.chunk_id,
            "source_system": result.source_system,
            "record_type": result.record_type,
            "title": result.title,
            "source_file": result.source_file,
            "citation_label": result.citation_label,
            "text_preview": result.text[: args.preview_chars],
        }
        for rank, result in enumerate(results, start=1)
    ]

    print_json(output)

    if not output:
        print("[WARN] No vector results found.")

    return 0


def command_hybrid_search(args: argparse.Namespace) -> int:
    """Run hybrid retrieval."""

    settings = load_config(args.config)

    retriever = HybridRetriever(settings)
    request = HybridSearchRequest(
        query=args.query,
        top_k=args.top_k,
        source_system=args.source_system,
        record_type=args.record_type,
    )
    response = retriever.search(request)

    print_json(response.to_dict(preview_chars=args.preview_chars))

    if not response.results:
        print("[WARN] No hybrid results found.")
        return 1

    print(f"[PASS] Hybrid search completed. Confidence: {response.confidence}")
    return 0



def command_rag_answer(args: argparse.Namespace) -> int:
    """Run grounded answer generation."""

    settings = load_config(args.config)

    async def _run() -> Any:
        generator = RAGAnswerGenerator(settings)
        request = RAGAnswerRequest(
            query=args.query,
            top_k=args.top_k,
            source_system=args.source_system,
            record_type=args.record_type,
            show_evidence=args.show_evidence,
        )
        return await generator.answer(request)

    response = asyncio.run(_run())
    print_json(
        response.to_dict(
            preview_chars=args.preview_chars,
            include_evidence=args.show_evidence,
        )
    )

    if response.status == "ok":
        print(f"[PASS] RAG answer generated. Confidence: {response.confidence}")
        return 0

    if response.status == "no_answer":
        print(f"[WARN] RAG answer returned no-answer fallback. Confidence: {response.confidence}")
        return 0

    print(f"[FAIL] RAG answer failed. Status: {response.status}")
    return 1


def find_first_image(input_dir: str | Path) -> Path | None:
    """Find first image under input directory."""

    root = Path(input_dir)

    if not root.exists():
        return None

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path

    return None


def command_test_ocr(args: argparse.Namespace) -> int:
    """Test OCR on one image."""

    settings = load_config(args.config)
    image_path = Path(args.image) if args.image else find_first_image(settings.paths.input_data)

    if image_path is None:
        print("[FAIL] No supported image found.")
        return 1

    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        return 1

    client = LocalOcrClient(settings.ocr)

    print(f"[INFO] OCR enabled: {settings.ocr.enabled}")
    print(f"[INFO] OCR provider: {settings.ocr.provider}")
    print(f"[INFO] Tesseract command: {settings.ocr.tesseract_cmd}")
    print(f"[INFO] Image: {image_path}")

    result = client.ocr_image_file(image_path)

    output = {
        "status": result.status,
        "image": str(image_path),
        "text": result.text,
        "error": result.error,
    }

    print_json(output)

    if result.status != "ok":
        print("[WARN] OCR did not produce usable text.")
        return 1

    print("[PASS] OCR test succeeded.")
    return 0


def command_test_vision(args: argparse.Namespace) -> int:
    """Test vision captioning on one image."""

    settings = load_config(args.config)
    image_path = Path(args.image) if args.image else find_first_image(settings.paths.input_data)

    if not settings.vision.enabled or settings.vision.mode != "caption":
        print("[FAIL] Vision captioning is not enabled in config.")
        return 1

    if image_path is None:
        print("[FAIL] No supported image found.")
        return 1

    if not image_path.exists():
        print(f"[FAIL] Image not found: {image_path}")
        return 1

    client = LocalVisionClient(settings.vision)

    print(f"[INFO] Vision provider: {settings.vision.provider}")
    print(f"[INFO] Vision model: {settings.vision.model}")
    print(f"[INFO] Image: {image_path}")

    result = client.caption_image_file(
        image_path=image_path,
        prompt=(
            "Describe this image for a retrieval system. "
            "Focus on visible objects, equipment, readable labels, company names, "
            "product names, logos, model numbers, and useful identifying details. "
            "If text is not readable, say so. Do not invent details."
        ),
    )

    output = {
        "status": result.status,
        "image": str(image_path),
        "model": settings.vision.model,
        "caption": result.caption,
        "error": result.error,
    }

    print_json(output)

    if result.status != "ok":
        print("[FAIL] Vision model test failed.")
        return 1

    print("[PASS] Vision model test succeeded.")
    return 0


def command_clean(args: argparse.Namespace) -> int:
    """Clean generated runtime outputs."""

    settings = load_config(args.config)

    processed_dir = Path(settings.paths.processed_data)
    metadata_db = Path(settings.paths.metadata_db)
    vector_dir = Path(settings.vector_index.index_dir)

    removed: list[str] = []

    if args.processed:
        for pattern in ("*.json", "*.jsonl"):
            for path in processed_dir.glob(pattern):
                path.unlink(missing_ok=True)
                removed.append(str(path))

    if args.metadata and metadata_db.exists():
        metadata_db.unlink()
        removed.append(str(metadata_db))

    if args.vector and vector_dir.exists():
        shutil.rmtree(vector_dir)
        removed.append(str(vector_dir))

    output = {
        "processed_cleaned": args.processed,
        "metadata_cleaned": args.metadata,
        "vector_cleaned": args.vector,
        "removed": removed,
    }

    print_json(output)
    print("[PASS] Clean completed.")
    return 0


def _clean_entrypoint(args: argparse.Namespace) -> int:
    """Normalize clean flags."""

    if args.all:
        args.processed = True
        args.metadata = True
        args.vector = True

    if not args.processed and not args.metadata and not args.vector:
        print("[FAIL] Nothing selected to clean. Use --processed, --metadata, --vector, or --all.")
        return 1

    return command_clean(args)


def command_rebuild(args: argparse.Namespace) -> int:
    """Rebuild ingestion and search indexes."""

    if args.fresh:
        clean_args = argparse.Namespace(
            config=args.config,
            processed=True,
            metadata=True,
            vector=True,
        )
        clean_code = command_clean(clean_args)
        if clean_code != 0:
            return clean_code

    if args.extract:
        extract_args = argparse.Namespace(config=args.config, clear=args.clear_extract)
        extract_code = command_extract(extract_args)
        if extract_code != 0:
            return extract_code

    ingest_code = command_ingest(argparse.Namespace(config=args.config, force=False, full_scan=False))
    if ingest_code != 0:
        return ingest_code

    count_code = command_count(argparse.Namespace(config=args.config))
    if count_code != 0:
        return count_code

    keyword_code = command_build_keyword_index(argparse.Namespace(config=args.config))
    if keyword_code != 0:
        return keyword_code

    index_code = command_build_index(argparse.Namespace(config=args.config))
    if index_code != 0:
        return index_code

    print("[PASS] Rebuild completed.")
    return 0


def command_status(args: argparse.Namespace) -> int:
    """Show compact system status."""

    settings = load_config(args.config)

    metadata_db = Path(settings.paths.metadata_db)
    vector_dir = Path(settings.vector_index.index_dir)
    embeddings_path = vector_dir / settings.vector_index.embeddings_file
    metadata_path = vector_dir / settings.vector_index.metadata_file

    db_status: dict[str, Any] = {
        "exists": metadata_db.exists(),
        "path": str(metadata_db),
    }

    if metadata_db.exists():
        with sqlite3.connect(metadata_db) as conn:
            db_status["documents"] = safe_count_table(conn, "documents")
            db_status["chunks"] = safe_count_table(conn, "chunks")
            db_status["ingestion_runs"] = safe_count_table(conn, "ingestion_runs")

            if table_exists_for_cli(conn, "chunks"):
                rows = conn.execute(
                    """
                    SELECT record_type, COUNT(*) AS count
                    FROM chunks
                    GROUP BY record_type
                    ORDER BY count DESC;
                    """
                ).fetchall()
            else:
                rows = []

            db_status["record_types"] = [
                {"record_type": row[0], "count": row[1]}
                for row in rows
            ]

    index_status: dict[str, Any] = {
        "index_dir": str(vector_dir),
        "embeddings_exists": embeddings_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }

    if embeddings_path.exists():
        vectors = np.load(embeddings_path)
        index_status["vector_shape"] = list(vectors.shape)

    keyword_status = KeywordSearchIndex(
        metadata_db_path=settings.paths.metadata_db,
        table_name=settings.retrieval.keyword_table,
    ).table_status()

    registry = FileRegistry(settings.paths.metadata_db)
    file_registry_status = registry.summary() if metadata_db.exists() else {
        "exists": False,
        "total_files_tracked": 0,
        "status_counts": {},
    }

    lifecycle_status = get_index_lifecycle_status(
        metadata_db_path=settings.paths.metadata_db,
        vector_index_dir=settings.vector_index.index_dir,
        embeddings_file=settings.vector_index.embeddings_file,
        metadata_file=settings.vector_index.metadata_file,
        keyword_table=settings.retrieval.keyword_table,
    )

    output = {
        "config": args.config,
        "deployment": {
            "mode": settings.deployment.mode,
            "host": settings.app.host,
            "port": settings.app.port,
            "offline_mode": settings.app.offline_mode,
        },
        "llm": {
            "provider": settings.llm.provider,
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
        },
        "ocr": {
            "enabled": settings.ocr.enabled,
            "provider": settings.ocr.provider,
            "ocr_images": settings.ocr.ocr_images,
            "ocr_pdf_pages": settings.ocr.ocr_pdf_pages,
            "max_images_per_run": settings.ocr.max_images_per_run,
            "max_pdf_pages_per_run": settings.ocr.max_pdf_pages_per_run,
        },
        "vision": {
            "enabled": settings.vision.enabled,
            "provider": settings.vision.provider,
            "base_url": settings.vision.base_url,
            "model": settings.vision.model,
            "caption_images": settings.vision.caption_images,
            "caption_pdf_pages": settings.vision.caption_pdf_pages,
            "max_images_per_run": settings.vision.max_images_per_run,
            "max_pdf_pages_per_run": settings.vision.max_pdf_pages_per_run,
        },
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model_name_or_path,
            "device": settings.embeddings.device,
            "batch_size": settings.embeddings.batch_size,
        },
        "retrieval": {
            "keyword_table": settings.retrieval.keyword_table,
            "vector_top_k": settings.retrieval.vector_top_k,
            "keyword_top_k": settings.retrieval.keyword_top_k,
            "final_top_k": settings.retrieval.final_top_k,
            "vector_weight": settings.retrieval.vector_weight,
            "keyword_weight": settings.retrieval.keyword_weight,
            "exact_match_boost": settings.retrieval.exact_match_boost,
            "deduplicate": settings.retrieval.deduplicate,
        },
        "rag": {
            "max_context_chunks": settings.rag.max_context_chunks,
            "max_context_chars": settings.rag.max_context_chars,
            "max_chars_per_chunk": settings.rag.max_chars_per_chunk,
            "drop_vector_only_below_score": settings.rag.drop_vector_only_below_score,
            "min_retrieval_confidence": settings.rag.min_retrieval_confidence,
            "require_citations": settings.rag.require_citations,
            "retry_on_invalid_citations": settings.rag.retry_on_invalid_citations,
            "answer_temperature": settings.rag.answer_temperature,
            "answer_max_tokens": settings.rag.answer_max_tokens,
        },
        "file_tracking": {
            "enabled": settings.file_tracking.enabled,
            "work_dir": settings.file_tracking.work_dir,
            "registry": file_registry_status,
        },
        "metadata_db": db_status,
        "keyword_index": keyword_status,
        "vector_index": index_status,
        "index_lifecycle": lifecycle_status,
    }

    print_json(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="RAGS PoC management CLI.")

    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inventory input files.")
    inventory.add_argument("--sample-limit", type=int, default=10)
    inventory.set_defaults(func=command_inventory)

    extract = subparsers.add_parser("extract", help="Extract archives.")
    extract.add_argument("--clear", action="store_true", help="Clear extract dir before extraction.")
    extract.set_defaults(func=command_extract)

    plan_ingest = subparsers.add_parser("plan-ingest", help="Show added/changed/deleted files without ingesting.")
    plan_ingest.set_defaults(func=command_plan_ingest)

    ingest = subparsers.add_parser("ingest", help="Run incremental ingestion by default.")
    ingest.add_argument("--force", action="store_true", help="Reprocess all supported files through incremental ingestion.")
    ingest.add_argument("--full-scan", action="store_true", help="Use legacy all-files ingestion pipeline.")
    ingest.set_defaults(func=command_ingest)

    count = subparsers.add_parser("count", help="Count chunk record types.")
    count.set_defaults(func=command_count)

    inspect_metadata = subparsers.add_parser("inspect-metadata", help="Inspect recent metadata chunks.")
    inspect_metadata.add_argument("--limit", type=int, default=20)
    inspect_metadata.add_argument("--preview-chars", type=int, default=300)
    inspect_metadata.set_defaults(func=command_inspect_metadata)

    build_keyword = subparsers.add_parser("build-keyword-index", help="Build SQLite FTS5 keyword index.")
    build_keyword.set_defaults(func=command_build_keyword_index)

    build_index = subparsers.add_parser("build-index", help="Build vector index.")
    build_index.set_defaults(func=command_build_index)

    inspect_index = subparsers.add_parser("inspect-index", help="Inspect vector index.")
    inspect_index.set_defaults(func=command_inspect_index)

    search = subparsers.add_parser("search", help="Search vector index only.")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--source-system", default=None)
    search.add_argument("--record-type", default=None)
    search.add_argument("--preview-chars", type=int, default=500)
    search.set_defaults(func=command_search)

    hybrid = subparsers.add_parser("hybrid-search", help="Run hybrid retrieval.")
    hybrid.add_argument("--query", required=True)
    hybrid.add_argument("--top-k", type=int, default=8)
    hybrid.add_argument("--source-system", default=None)
    hybrid.add_argument("--record-type", default=None)
    hybrid.add_argument("--preview-chars", type=int, default=600)
    hybrid.set_defaults(func=command_hybrid_search)

    rag_answer = subparsers.add_parser("rag-answer", help="Run grounded answer generation.")
    rag_answer.add_argument("--query", required=True)
    rag_answer.add_argument("--top-k", type=int, default=8)
    rag_answer.add_argument("--source-system", default=None)
    rag_answer.add_argument("--record-type", default=None)
    rag_answer.add_argument("--show-evidence", action="store_true", help="Include evidence previews in the JSON response.")
    rag_answer.add_argument("--preview-chars", type=int, default=700)
    rag_answer.set_defaults(func=command_rag_answer)

    test_ocr = subparsers.add_parser("test-ocr", help="Test OCR on one image.")
    test_ocr.add_argument("--image", default=None)
    test_ocr.set_defaults(func=command_test_ocr)

    test_vision = subparsers.add_parser("test-vision", help="Test vision on one image.")
    test_vision.add_argument("--image", default=None)
    test_vision.set_defaults(func=command_test_vision)

    clean = subparsers.add_parser("clean", help="Clean generated runtime outputs.")
    clean.add_argument("--processed", action="store_true", help="Remove processed JSON/JSONL files.")
    clean.add_argument("--metadata", action="store_true", help="Remove metadata DB.")
    clean.add_argument("--vector", action="store_true", help="Remove vector index folder.")
    clean.add_argument("--all", action="store_true", help="Remove processed, metadata, and vector outputs.")
    clean.set_defaults(func=_clean_entrypoint)

    rebuild = subparsers.add_parser("rebuild", help="Run ingest + keyword index + vector index.")
    rebuild.add_argument("--fresh", action="store_true", help="Clean processed/metadata/vector before rebuild.")
    rebuild.add_argument("--extract", action="store_true", help="Run archive extraction before ingestion.")
    rebuild.add_argument("--clear-extract", action="store_true", help="Clear extracted archive folder before extraction.")
    rebuild.set_defaults(func=command_rebuild)

    status = subparsers.add_parser("status", help="Show compact system status.")
    status.set_defaults(func=command_status)

    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
