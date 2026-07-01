"""Configuration model and loader.

Purpose
-------
Defines the full, validated settings model for Ariadne and loads it from YAML so
every component reads configuration from one typed, predictable place.

What it does
------------
Declares typed sections for deployment, the local LLM, paths, ingestion,
vision/OCR, embeddings, the vector and keyword indexes, retrieval, the metadata
signal, answer generation, runtime concurrency, and security. Each section
validates its own values (ranges, allowed choices, positive numbers).

Flow
----
``load_settings()`` reads the base config file, deep-merges any saved UI overrides
on top, and constructs the validated ``Settings`` object. Invalid values fail
fast with a clear error at load time rather than at runtime.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DeploymentConfig(BaseModel):
    """
    Deployment mode for the PoC.

    local:
        Backend is intended for same-machine use.
        Typical host: 127.0.0.1

    lan:
        Backend is intended to be reachable by other machines on the LAN.
        Typical host: 0.0.0.0
    """

    mode: str = "local"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        supported = {"local", "lan"}
        if value not in supported:
            raise ValueError(f"Unsupported deployment mode '{value}'. Supported: {supported}")
        return value


class AppConfig(BaseModel):
    """
    Application-level runtime settings.
    """

    name: str = "RAGS PoC"
    host: str = "127.0.0.1"
    port: int = 8080
    offline_mode: bool = True


class LLMConfig(BaseModel):
    """
    Local LLM connection settings.

    Default integration contract:
        OpenAI-compatible local HTTP endpoint.

    This keeps Llama 3 or any other local model swappable by config.
    """

    provider: str = "openai_compatible"
    base_url: str = "http://localhost:11434/v1"
    model: str = "test-model"
    temperature: float = 0.1
    max_tokens: int = 700
    timeout_seconds: int = 120

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        supported = {"openai_compatible"}
        if value not in supported:
            raise ValueError(f"Unsupported LLM provider '{value}'. Supported: {supported}")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_tokens must be greater than 0")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value


class PathsConfig(BaseModel):
    """
    Local filesystem paths.
    """

    input_data: str = "data/input"
    processed_data: str = "data/processed"
    metadata_db: str = "storage/metadata.db"
    logs: str = "storage/logs"


class AutoMetadataConfig(BaseModel):
    """Auto-metadata: the local LLM extracts config-defined fields at ingest."""

    enabled: bool = False
    sample_chars: int = 4000
    fields: list[dict] = []


class IngestionConfig(BaseModel):
    """
    Ingestion settings shared by all loaders.
    """

    max_chars: int = 1200
    overlap_chars: int = 150
    auto_metadata: AutoMetadataConfig = AutoMetadataConfig()

    @field_validator("max_chars")
    @classmethod
    def validate_max_chars(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_chars must be greater than 0")
        return value

    @field_validator("overlap_chars")
    @classmethod
    def validate_overlap_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("overlap_chars cannot be negative")
        return value


class VisionConfig(BaseModel):
    """
    Image and image-only PDF handling settings.

    Default behavior:
        Images and scanned/image-only PDF pages are catalogued only.

    Optional behavior:
        If a local vision model is available, set enabled=true and mode=caption.

    Runtime controls:
        Captioning can be slow. The per-run limits prevent accidental long
        ingestion runs over hundreds of images/pages.
    """

    enabled: bool = False
    mode: str = "catalog"
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llava:7b"
    timeout_seconds: int = 180
    caption_images: bool = False
    caption_pdf_pages: bool = False
    max_images_per_run: int = 25
    max_pdf_pages_per_run: int = 25

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        supported = {"catalog", "caption"}
        if value not in supported:
            raise ValueError(f"Unsupported vision mode '{value}'. Supported: {supported}")
        return value

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        supported = {"ollama"}
        if value not in supported:
            raise ValueError(f"Unsupported vision provider '{value}'. Supported: {supported}")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value

    @field_validator("max_images_per_run")
    @classmethod
    def validate_max_images(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_images_per_run cannot be negative")
        return value

    @field_validator("max_pdf_pages_per_run")
    @classmethod
    def validate_max_pdf_pages(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_pdf_pages_per_run cannot be negative")
        return value


class ArchivesConfig(BaseModel):
    """
    Archive handling settings.

    Archives are not embedded directly. They should be extracted first, and
    extracted files then pass through the normal unified ingestion pipeline.
    """

    enabled: bool = True
    extract_dir: str = "data/input/_extracted"
    keep_original_metadata: bool = True


class SecurityConfig(BaseModel):
    """
    Basic PoC safety flags.
    """

    allow_external_calls: bool = False
    log_full_prompts: bool = False
    approved_lan_hosts: list[str] = Field(default_factory=list)


class EmbeddingsConfig(BaseModel):
    """
    Local embedding model settings.

    provider:
        Currently only sentence_transformers is supported.

    model_name_or_path:
        Can be a Hugging Face model name during development or a local model
        folder path for offline/client deployment.

    device:
        cuda, cpu, or auto.

    batch_size:
        Number of chunks embedded per batch.
    """

    provider: str = "sentence_transformers"
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda"
    batch_size: int = 32

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        supported = {"sentence_transformers"}
        if value not in supported:
            raise ValueError(f"Unsupported embedding provider '{value}'. Supported: {supported}")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        supported = {"cuda", "cpu", "auto"}
        if value not in supported:
            raise ValueError(f"Unsupported embedding device '{value}'. Supported: {supported}")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size must be greater than 0")
        return value


class VectorIndexConfig(BaseModel):
    """
    Local vector index settings.

    The PoC uses a lightweight NumPy vector index instead of a vector database.
    This keeps installation simple and avoids extra services.
    """

    index_dir: str = "storage/vector"
    embeddings_file: str = "embeddings.npy"
    metadata_file: str = "metadata.jsonl"


class MetadataSignalConfig(BaseModel):
    """Metadata leg of retrieval: soft document-type signal."""

    enabled: bool = False
    query_intent_classification: bool = True
    document_types: list[str] = []


class RetrievalConfig(BaseModel):
    """
    Hybrid retrieval settings.

    Retrieval uses local vector search plus SQLite keyword search. These values
    are intentionally configurable because tuning retrieval is expected during
    PoC validation and later UI/admin controls.
    """

    keyword_table: str = "chunks_fts"
    vector_top_k: int = 25
    keyword_top_k: int = 25
    final_top_k: int = 8
    min_score: float = 0.12
    aggregation_table_completion: bool = True
    aggregation_max_rows: int = 400
    # Multi-hop retrieval (Phase 0/1: decompose-then-retrieve). Off by default;
    # when off, behaviour is exactly single-pass. See ARIADNE_MULTIHOP_PLAN.
    multihop_enabled: bool = False
    multihop_mode: str = "decompose"        # "decompose" (Phase 1) | "iterative" (Phase 3)
    multihop_max_subquestions: int = 4
    multihop_max_hops: int = 3
    multihop_min_trigger: str = "auto"       # "auto" (heuristic gate) | "always"
    multihop_per_subq_top_k: int = 6
    vector_weight: float = 0.60
    keyword_weight: float = 0.40
    exact_match_boost: float = 0.20
    fusion_method: str = "rrf"
    rrf_k: int = 60
    confidence_high_score: float = 0.55
    confidence_medium_score: float = 0.35
    query_expansions: dict[str, str] = {}
    metadata_signal: MetadataSignalConfig = MetadataSignalConfig()
    document_type_nudge_weight: float = 0.15
    rerank_enabled: bool = False
    rerank_model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_pool: int = 12
    rerank_device: str = "auto"
    deduplicate: bool = True
    record_type_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "pdf_page": 1.00,
            "csv_row": 1.00,
            "xlsx_row": 1.00,
            "docx_section": 0.95,
            "pptx_slide": 0.95,
            "text_document": 1.00,
            "pdf_page_ocr_text": 0.90,
            "image_ocr_text": 0.85,
            "pdf_page_vision_caption": 0.75,
            "image_caption": 0.70,
            "pdf_page_image_metadata": 0.35,
            "image_metadata": 0.30,
        }
    )

    @field_validator("vector_top_k", "keyword_top_k", "final_top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("retrieval top-k values must be greater than 0")
        return value

    @field_validator("min_score")
    @classmethod
    def validate_min_score(cls, value: float) -> float:
        if value < 0:
            raise ValueError("min_score cannot be negative")
        return value

    @field_validator("vector_weight", "keyword_weight", "exact_match_boost")
    @classmethod
    def validate_weights(cls, value: float) -> float:
        if value < 0:
            raise ValueError("retrieval weights cannot be negative")
        return value

    @field_validator("fusion_method")
    @classmethod
    def validate_fusion_method(cls, value: str) -> str:
        normalized = (value or "weighted").lower().strip()
        supported = {"weighted", "rrf"}
        if normalized not in supported:
            raise ValueError(f"Unsupported fusion_method '{value}'. Supported: {supported}")
        return normalized

    @field_validator("rrf_k", "rerank_candidate_pool")
    @classmethod
    def validate_positive_retrieval_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("retrieval integer values must be greater than 0")
        return value

    @field_validator("rerank_device")
    @classmethod
    def validate_rerank_device(cls, value: str) -> str:
        normalized = (value or "auto").lower().strip()
        supported = {"auto", "cpu", "cuda"}
        if normalized not in supported:
            raise ValueError(f"Unsupported rerank_device '{value}'. Supported: {supported}")
        return normalized


class FileTrackingConfig(BaseModel):
    """
    Incremental ingestion settings.

    When enabled, normal ingestion processes only new/changed files, removes
    chunks for deleted files, and treats keyword/vector indexes as derived
    artifacts that should be rebuilt after changes.
    """

    enabled: bool = True
    work_dir: str = "storage/incremental_work"
    track_unsupported: bool = True
    auto_rebuild_keyword_index: bool = True
    auto_rebuild_vector_index: bool = True


class RAGConfig(BaseModel):
    """
    Grounded answer-generation settings.

    These controls define how much evidence is passed to the local LLM, how
    strict citation validation should be, and when the system should refuse to
    answer due to weak retrieval.
    """

    max_context_chunks: int = 6
    max_context_chars: int = 8000
    max_context_chars_table: int = 24000
    max_chars_per_chunk: int = 1800
    drop_vector_only_below_score: float = 0.30
    min_retrieval_confidence: str = "medium"
    citation_salvage_enabled: bool = True
    citation_salvage_min_overlap: float = 0.5
    require_citations: bool = True
    allow_no_answer: bool = True
    retry_on_invalid_citations: bool = True
    answer_temperature: float = 0.1
    answer_max_tokens: int = 700
    no_answer_message: str = "The indexed data does not contain enough information to answer this reliably."
    evidence_span_enabled: bool = True
    evidence_span_window_sentences: int = 2
    evidence_span_max_chars: int = 900
    deterministic_tables_enabled: bool = True
    deterministic_table_max_rows: int = 40
    claim_verification_enabled: bool = True
    claim_verification_min_overlap: float = 0.18

    @field_validator("max_context_chunks", "max_context_chars", "max_chars_per_chunk", "answer_max_tokens", "evidence_span_window_sentences", "evidence_span_max_chars", "deterministic_table_max_rows")
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("RAG integer limits must be greater than 0")
        return value

    @field_validator("min_retrieval_confidence")
    @classmethod
    def validate_min_retrieval_confidence(cls, value: str) -> str:
        normalized = value.lower().strip()
        supported = {"low", "medium", "high"}
        if normalized not in supported:
            raise ValueError(f"Unsupported min_retrieval_confidence '{value}'. Supported: {supported}")
        return normalized

    @field_validator("drop_vector_only_below_score", "claim_verification_min_overlap")
    @classmethod
    def validate_rag_float_bounds(cls, value: float) -> float:
        if value < 0:
            raise ValueError("RAG float values cannot be negative")
        return value

    @field_validator("answer_temperature")
    @classmethod
    def validate_answer_temperature(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("answer_temperature must be between 0 and 2")
        return value


class RuntimeConfig(BaseModel):
    """
    Local/LAN runtime concurrency settings.

    These limits keep the fully-local model stack responsive when multiple LAN
    users access the app. They do not create external services or cloud calls.
    """

    max_chat_concurrency: int = 3
    max_search_concurrency: int = 6
    max_admin_jobs: int = 1
    reject_chat_during_rebuild: bool = False

    @field_validator("max_chat_concurrency", "max_search_concurrency", "max_admin_jobs")
    @classmethod
    def validate_positive_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("runtime concurrency limits must be greater than 0")
        return value

class OcrConfig(BaseModel):
    """
    Local OCR settings.

    OCR is used after native text extraction and before vision captioning.

    Intended use:
        - scanned/image-only PDF pages
        - product cards
        - business cards
        - labels
        - screenshots
        - images with visible text
    """

    enabled: bool = False
    provider: str = "tesseract"
    tesseract_cmd: str | None = None
    languages: str = "eng"
    min_text_chars: int = 20
    psm: int = 6
    ocr_images: bool = True
    ocr_pdf_pages: bool = True
    max_images_per_run: int = 50
    max_pdf_pages_per_run: int = 50

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        supported = {"tesseract"}
        if value not in supported:
            raise ValueError(f"Unsupported OCR provider '{value}'. Supported: {supported}")
        return value

    @field_validator("min_text_chars")
    @classmethod
    def validate_min_text_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("min_text_chars cannot be negative")
        return value

    @field_validator("psm")
    @classmethod
    def validate_psm(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("psm must be greater than 0")
        return value

    @field_validator("max_images_per_run")
    @classmethod
    def validate_max_images(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_images_per_run cannot be negative")
        return value

    @field_validator("max_pdf_pages_per_run")
    @classmethod
    def validate_max_pdf_pages(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_pdf_pages_per_run cannot be negative")
        return value

class Settings(BaseModel):
    """
    Root settings object.
    """

    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    archives: ArchivesConfig = Field(default_factory=ArchivesConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_index: VectorIndexConfig = Field(default_factory=VectorIndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    file_tracking: FileTrackingConfig = Field(default_factory=FileTrackingConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override values into a base configuration dictionary.

    This is used for UI-safe configuration overrides. The base config remains
    config/client.yaml, while UI edits are written to config/ui_overrides.yaml.
    Keeping the files separate preserves comments and makes handover safer.
    """

    merged = dict(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must contain a mapping at the root: {path}")

    return data


def load_settings(
    config_path: str | Path = "config/client.yaml",
    overrides_path: str | Path | None = None,
) -> Settings:
    """
    Load application settings from YAML.

    Load order:
        1. config/client.yaml
        2. config/ui_overrides.yaml, if present

    The UI writes only safe tuning settings to ui_overrides.yaml. This avoids
    rewriting the main client.yaml file and keeps the system fully local.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_yaml_dict(path)

    if overrides_path is None:
        override_path = path.with_name("ui_overrides.yaml")
    else:
        override_path = Path(overrides_path)

    if override_path.exists():
        override_raw = _load_yaml_dict(override_path)
        raw = _deep_merge(raw, override_raw)

    return Settings(**raw)
