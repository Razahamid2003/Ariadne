"""Safe UI configuration overrides.

Purpose
-------
Lets the settings drawer change a curated set of tuning values safely, writing them
to an overrides file rather than editing the base configuration.

What it does
------------
Defines which fields are editable and their valid ranges, reads and writes the
overrides file, validates candidate values, and applies them on top of the
baseline.

Flow
----
The UI requests the editable schema, submits new values, and these are validated
and merged over the base config so changes take effect without touching the main
configuration file.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from backend.app.core.config import Settings, load_settings
from backend.app.services.network_safety import assert_allowed_endpoint


@dataclass(frozen=True)
class ConfigFieldSpec:
    """One UI-editable configuration field."""

    path: str
    label: str
    kind: str
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    help: str = ""
    choices: tuple[str, ...] = ()
    placeholder: str = ""
    local_url: bool = False

    def to_dict(self, current_value: Any) -> dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "kind": self.kind,
            "min": self.minimum,
            "max": self.maximum,
            "step": self.step,
            "value": current_value,
            "help": self.help,
            "choices": list(self.choices),
            "placeholder": self.placeholder,
            "local_url": self.local_url,
        }


SAFE_FIELDS: dict[str, ConfigFieldSpec] = {
    # Local model controls.
    "llm.base_url": ConfigFieldSpec(
        path="llm.base_url",
        label="Text model server",
        kind="text",
        local_url=True,
        placeholder="http://localhost:11434/v1",
        help="OpenAI-compatible local/LAN endpoint. Blocked if public while external calls are disabled.",
    ),
    "llm.model": ConfigFieldSpec(
        path="llm.model",
        label="Text model name",
        kind="text",
        placeholder="llama3.1:8b",
        help="The local model used for answer generation.",
    ),
    "llm.temperature": ConfigFieldSpec(
        path="llm.temperature",
        label="Model creativity",
        kind="float",
        minimum=0,
        maximum=2,
        step=0.05,
        help="Lower values are more deterministic and better for grounded RAG answers.",
    ),
    "llm.max_tokens": ConfigFieldSpec(
        path="llm.max_tokens",
        label="Model output limit",
        kind="int",
        minimum=64,
        maximum=4096,
        step=64,
        help="Maximum answer tokens requested from the local model.",
    ),
    "llm.timeout_seconds": ConfigFieldSpec(
        path="llm.timeout_seconds",
        label="Text model timeout",
        kind="int",
        minimum=15,
        maximum=900,
        step=15,
        help="Longer timeouts help large local models finish RAG prompts.",
    ),

    # Ingestion/chunking controls.
    "paths.input_data": ConfigFieldSpec(
        path="paths.input_data",
        label="Input files folder",
        kind="text",
        placeholder="data/input",
        help="Local folder scanned during ingestion.",
    ),
    "ingestion.max_chars": ConfigFieldSpec(
        path="ingestion.max_chars",
        label="Chunk size",
        kind="int",
        minimum=300,
        maximum=5000,
        step=100,
        help="Maximum characters in each extracted text chunk.",
    ),
    "ingestion.overlap_chars": ConfigFieldSpec(
        path="ingestion.overlap_chars",
        label="Chunk overlap",
        kind="int",
        minimum=0,
        maximum=1000,
        step=25,
        help="Characters repeated between neighboring chunks for continuity.",
    ),
    "archives.enabled": ConfigFieldSpec(
        path="archives.enabled",
        label="Extract archives",
        kind="bool",
        help="Enable archive extraction before ingestion.",
    ),
    "archives.extract_dir": ConfigFieldSpec(
        path="archives.extract_dir",
        label="Archive extraction folder",
        kind="text",
        placeholder="data/input/_extracted",
        help="Where ZIP/archive contents are extracted locally.",
    ),
    "archives.keep_original_metadata": ConfigFieldSpec(
        path="archives.keep_original_metadata",
        label="Keep archive provenance",
        kind="bool",
        help="Preserve original archive/source metadata for extracted files.",
    ),

    # OCR fallback controls.
    "ocr.enabled": ConfigFieldSpec(path="ocr.enabled", label="Use OCR fallback", kind="bool", help="Run OCR after native text extraction is weak."),
    "ocr.tesseract_cmd": ConfigFieldSpec(path="ocr.tesseract_cmd", label="Tesseract path", kind="text", placeholder="C:/Program Files/Tesseract-OCR/tesseract.exe", help="Optional local Tesseract executable path."),
    "ocr.languages": ConfigFieldSpec(path="ocr.languages", label="OCR languages", kind="text", placeholder="eng", help="Tesseract language codes, for example eng."),
    "ocr.min_text_chars": ConfigFieldSpec(path="ocr.min_text_chars", label="Minimum native text before OCR skip", kind="int", minimum=0, maximum=1000, step=5, help="If native extraction gets this much text, OCR can be skipped."),
    "ocr.psm": ConfigFieldSpec(path="ocr.psm", label="OCR page segmentation mode", kind="int", minimum=1, maximum=13, step=1, help="Tesseract PSM value."),
    "ocr.ocr_images": ConfigFieldSpec(path="ocr.ocr_images", label="OCR image files", kind="bool", help="Use OCR on standalone images."),
    "ocr.ocr_pdf_pages": ConfigFieldSpec(path="ocr.ocr_pdf_pages", label="OCR PDF pages", kind="bool", help="Use OCR on scanned/image-heavy PDF pages."),
    "ocr.max_images_per_run": ConfigFieldSpec(path="ocr.max_images_per_run", label="OCR image limit per run", kind="int", minimum=0, maximum=5000, step=5, help="Maximum image files OCR will process in one ingestion run."),
    "ocr.max_pdf_pages_per_run": ConfigFieldSpec(path="ocr.max_pdf_pages_per_run", label="OCR PDF page limit per run", kind="int", minimum=0, maximum=10000, step=10, help="Maximum PDF pages OCR will process in one ingestion run."),

    # Vision fallback controls.
    "vision.enabled": ConfigFieldSpec(path="vision.enabled", label="Use vision fallback", kind="bool", help="Enable local vision captioning after native text/OCR are weak."),
    "vision.mode": ConfigFieldSpec(path="vision.mode", label="Vision mode", kind="select", choices=("catalog", "caption"), help="Catalog only stores metadata; caption sends images/pages to a local vision model."),
    "vision.base_url": ConfigFieldSpec(path="vision.base_url", label="Vision model server", kind="text", local_url=True, placeholder="http://localhost:11434", help="Local/LAN vision endpoint. Public endpoints are blocked in local-only mode."),
    "vision.model": ConfigFieldSpec(path="vision.model", label="Vision model name", kind="text", placeholder="qwen2.5vl:7b", help="Local vision model used for image/page captions."),
    "vision.timeout_seconds": ConfigFieldSpec(path="vision.timeout_seconds", label="Vision timeout", kind="int", minimum=15, maximum=900, step=15, help="Maximum seconds for a local vision request."),
    "vision.caption_images": ConfigFieldSpec(path="vision.caption_images", label="Caption image files", kind="bool", help="Caption standalone image files with the local vision model."),
    "vision.caption_pdf_pages": ConfigFieldSpec(path="vision.caption_pdf_pages", label="Caption PDF pages", kind="bool", help="Caption image-heavy PDF pages with the local vision model."),
    "vision.max_images_per_run": ConfigFieldSpec(path="vision.max_images_per_run", label="Vision image limit per run", kind="int", minimum=0, maximum=5000, step=5, help="Maximum image captions in one ingestion run."),
    "vision.max_pdf_pages_per_run": ConfigFieldSpec(path="vision.max_pdf_pages_per_run", label="Vision PDF page limit per run", kind="int", minimum=0, maximum=10000, step=10, help="Maximum PDF page captions in one ingestion run."),

    # Embeddings/index lifecycle.
    "embeddings.model_name_or_path": ConfigFieldSpec(path="embeddings.model_name_or_path", label="Embedding model path/name", kind="text", placeholder="sentence-transformers/all-MiniLM-L6-v2", help="Use a local path for fully offline packaging later."),
    "embeddings.device": ConfigFieldSpec(path="embeddings.device", label="Embedding device", kind="select", choices=("auto", "cpu", "cuda"), help="Device used for local embedding generation."),
    "embeddings.batch_size": ConfigFieldSpec(path="embeddings.batch_size", label="Embedding batch size", kind="int", minimum=1, maximum=512, step=1, help="Chunks embedded per batch during vector index rebuild."),
    "file_tracking.enabled": ConfigFieldSpec(path="file_tracking.enabled", label="Incremental ingestion", kind="bool", help="Track new/changed/deleted files instead of reprocessing everything."),
    "file_tracking.work_dir": ConfigFieldSpec(path="file_tracking.work_dir", label="File-tracking work folder", kind="text", placeholder="storage/incremental_work", help="Local folder for incremental ingestion state."),
    "file_tracking.track_unsupported": ConfigFieldSpec(path="file_tracking.track_unsupported", label="Track unsupported files", kind="bool", help="Remember unsupported files in the registry."),
    "file_tracking.auto_rebuild_keyword_index": ConfigFieldSpec(path="file_tracking.auto_rebuild_keyword_index", label="Auto-refresh exact search", kind="bool", help="Rebuild keyword/FTS index after ingestion changes."),
    "file_tracking.auto_rebuild_vector_index": ConfigFieldSpec(path="file_tracking.auto_rebuild_vector_index", label="Auto-refresh vector search", kind="bool", help="Rebuild vector index after ingestion changes."),

    # Retrieval controls.
    "retrieval.final_top_k": ConfigFieldSpec(path="retrieval.final_top_k", label="Source excerpts returned", kind="int", minimum=1, maximum=50, step=1, help="How many ranked source excerpts retrieval returns."),
    "retrieval.vector_top_k": ConfigFieldSpec(path="retrieval.vector_top_k", label="Meaning-search candidates", kind="int", minimum=1, maximum=200, step=1, help="How many vector candidates are considered before final ranking."),
    "retrieval.keyword_top_k": ConfigFieldSpec(path="retrieval.keyword_top_k", label="Exact-match candidates", kind="int", minimum=1, maximum=200, step=1, help="How many keyword/FTS candidates are considered before final ranking."),
    "retrieval.vector_weight": ConfigFieldSpec(path="retrieval.vector_weight", label="Meaning-search influence", kind="float", minimum=0, maximum=1, step=0.05, help="Vector similarity contribution to ranking."),
    "retrieval.keyword_weight": ConfigFieldSpec(path="retrieval.keyword_weight", label="Exact-match influence", kind="float", minimum=0, maximum=1, step=0.05, help="Keyword/FTS contribution to ranking."),
    "retrieval.exact_match_boost": ConfigFieldSpec(path="retrieval.exact_match_boost", label="Exact ID/code boost", kind="float", minimum=0, maximum=2, step=0.05, help="Extra score for exact IDs, codes, and model names found in evidence."),
    "retrieval.min_score": ConfigFieldSpec(path="retrieval.min_score", label="Minimum source quality", kind="float", minimum=0, maximum=1, step=0.01, help="Hide retrieved excerpts below this score."),
    "retrieval.deduplicate": ConfigFieldSpec(path="retrieval.deduplicate", label="Deduplicate retrieved chunks", kind="bool", help="Remove duplicate or repeated source excerpts from the trail."),
    "retrieval.fusion_method": ConfigFieldSpec(path="retrieval.fusion_method", label="Rank weaving method", kind="select", choices=("rrf", "weighted"), help="Choose how meaning-search and exact-match trails are merged."),
    "retrieval.rrf_k": ConfigFieldSpec(path="retrieval.rrf_k", label="Rank-weaving balance", kind="int", minimum=10, maximum=120, step=5, help="Balances the two search trails when rank weaving is selected."),
    "retrieval.rerank_enabled": ConfigFieldSpec(path="retrieval.rerank_enabled", label="Second reading pass", kind="bool", help="Let Ariadne reread candidate excerpts before choosing the final trail."),
    "retrieval.rerank_model_name_or_path": ConfigFieldSpec(path="retrieval.rerank_model_name_or_path", label="Rereader model", kind="text", placeholder="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Local reranker model name or path used for the second reading pass."),
    "retrieval.rerank_candidate_pool": ConfigFieldSpec(path="retrieval.rerank_candidate_pool", label="Second-pass evidence pool", kind="int", minimum=8, maximum=100, step=1, help="How many retrieved excerpts Ariadne may reread before choosing sources."),
    "retrieval.rerank_device": ConfigFieldSpec(path="retrieval.rerank_device", label="Rereader device", kind="select", choices=("auto", "cpu", "cuda"), help="Device used by the local reranker."),

    # RAG answer controls.
    "rag.max_context_chunks": ConfigFieldSpec(path="rag.max_context_chunks", label="Sources sent to model", kind="int", minimum=1, maximum=30, step=1, help="Maximum source excerpts passed to the local LLM."),
    "rag.max_context_chars": ConfigFieldSpec(path="rag.max_context_chars", label="Total evidence length", kind="int", minimum=1000, maximum=60000, step=500, help="Maximum total evidence characters sent to the model."),
    "rag.max_chars_per_chunk": ConfigFieldSpec(path="rag.max_chars_per_chunk", label="Length per source excerpt", kind="int", minimum=300, maximum=10000, step=100, help="Maximum text copied from any one retrieved source."),
    "rag.drop_vector_only_below_score": ConfigFieldSpec(path="rag.drop_vector_only_below_score", label="Hide weak meaning-only matches", kind="float", minimum=0, maximum=1, step=0.05, help="Filters weak semantic-only matches before generation."),
    "rag.min_retrieval_confidence": ConfigFieldSpec(path="rag.min_retrieval_confidence", label="Minimum retrieval confidence", kind="select", choices=("low", "medium", "high"), help="Refusal threshold based on retrieval strength."),
    "rag.require_citations": ConfigFieldSpec(path="rag.require_citations", label="Require citations", kind="bool", help="Factual answers must cite retrieved evidence."),
    "rag.allow_no_answer": ConfigFieldSpec(path="rag.allow_no_answer", label="Allow no-answer fallback", kind="bool", help="Return a safe refusal when evidence is insufficient."),
    "rag.retry_on_invalid_citations": ConfigFieldSpec(path="rag.retry_on_invalid_citations", label="Retry invalid citations", kind="bool", help="Retry once if the model cites labels not in retrieved evidence."),
    "rag.answer_temperature": ConfigFieldSpec(path="rag.answer_temperature", label="Answer creativity", kind="float", minimum=0, maximum=2, step=0.05, help="Temperature used for RAG answer generation."),
    "rag.answer_max_tokens": ConfigFieldSpec(path="rag.answer_max_tokens", label="Answer token limit", kind="int", minimum=64, maximum=4096, step=64, help="Token limit for generated answers."),
    "rag.evidence_span_enabled": ConfigFieldSpec(path="rag.evidence_span_enabled", label="Carry focused passages", kind="bool", help="Carry only the most relevant passage from each source into the answer loom."),
    "rag.evidence_span_window_sentences": ConfigFieldSpec(path="rag.evidence_span_window_sentences", label="Nearby lines kept", kind="int", minimum=1, maximum=8, step=1, help="How many neighboring lines travel with a matched passage."),
    "rag.evidence_span_max_chars": ConfigFieldSpec(path="rag.evidence_span_max_chars", label="Focused passage length", kind="int", minimum=200, maximum=4000, step=50, help="Maximum length of each focused source passage."),
    "rag.deterministic_tables_enabled": ConfigFieldSpec(path="rag.deterministic_tables_enabled", label="Shape evidence tables", kind="bool", help="Format retrieved table rows directly when the question asks for a table."),
    "rag.deterministic_table_max_rows": ConfigFieldSpec(path="rag.deterministic_table_max_rows", label="Rows in evidence tables", kind="int", minimum=5, maximum=150, step=5, help="Maximum retrieved rows formatted into a table."),
    "rag.claim_verification_enabled": ConfigFieldSpec(path="rag.claim_verification_enabled", label="Check the answer thread", kind="bool", help="Review cited claims against their source passages before returning the answer."),
    "rag.claim_verification_min_overlap": ConfigFieldSpec(path="rag.claim_verification_min_overlap", label="Source support strictness", kind="float", minimum=0.05, maximum=0.60, step=0.01, help="How much source wording should support each cited claim."),

    # Runtime and LAN safety.
    "runtime.max_chat_concurrency": ConfigFieldSpec(path="runtime.max_chat_concurrency", label="Simultaneous answer requests", kind="int", minimum=1, maximum=20, step=1, help="How many LAN users can generate answers at once."),
    "runtime.max_search_concurrency": ConfigFieldSpec(path="runtime.max_search_concurrency", label="Simultaneous source searches", kind="int", minimum=1, maximum=50, step=1, help="How many retrieval-only searches can run at once."),
    "runtime.max_admin_jobs": ConfigFieldSpec(path="runtime.max_admin_jobs", label="Simultaneous admin jobs", kind="int", minimum=1, maximum=5, step=1, help="How many ingestion/rebuild jobs can run at once."),
    "runtime.reject_chat_during_rebuild": ConfigFieldSpec(path="runtime.reject_chat_during_rebuild", label="Reject chat during rebuild", kind="bool", help="When on, users cannot query while indexes are being rebuilt."),
    "security.approved_lan_hosts": ConfigFieldSpec(path="security.approved_lan_hosts", label="Approved LAN hostnames", kind="csv", placeholder="modelbox.local, ariadne-gpu", help="Optional explicit internal hostnames for model endpoints. Prefer private IPs to avoid DNS leakage."),
}


GROUPS: dict[str, list[str]] = {
    "Local text model": ["llm.base_url", "llm.model", "llm.temperature", "llm.max_tokens", "llm.timeout_seconds"],
    "Ingestion and chunking": ["paths.input_data", "ingestion.max_chars", "ingestion.overlap_chars", "archives.enabled", "archives.extract_dir", "archives.keep_original_metadata"],
    "OCR fallback": ["ocr.enabled", "ocr.tesseract_cmd", "ocr.languages", "ocr.min_text_chars", "ocr.psm", "ocr.ocr_images", "ocr.ocr_pdf_pages", "ocr.max_images_per_run", "ocr.max_pdf_pages_per_run"],
    "Vision fallback": ["vision.enabled", "vision.mode", "vision.base_url", "vision.model", "vision.timeout_seconds", "vision.caption_images", "vision.caption_pdf_pages", "vision.max_images_per_run", "vision.max_pdf_pages_per_run"],
    "Embeddings and index lifecycle": ["embeddings.model_name_or_path", "embeddings.device", "embeddings.batch_size", "file_tracking.enabled", "file_tracking.work_dir", "file_tracking.track_unsupported", "file_tracking.auto_rebuild_keyword_index", "file_tracking.auto_rebuild_vector_index"],
    "Search behavior": ["retrieval.final_top_k", "retrieval.vector_top_k", "retrieval.keyword_top_k", "retrieval.fusion_method", "retrieval.vector_weight", "retrieval.keyword_weight", "retrieval.exact_match_boost", "retrieval.min_score", "retrieval.deduplicate", "retrieval.rrf_k", "retrieval.rerank_enabled", "retrieval.rerank_candidate_pool", "retrieval.rerank_model_name_or_path", "retrieval.rerank_device"],
    "Answer generation": ["rag.max_context_chunks", "rag.max_context_chars", "rag.max_chars_per_chunk", "rag.drop_vector_only_below_score", "rag.min_retrieval_confidence", "rag.require_citations", "rag.allow_no_answer", "rag.retry_on_invalid_citations", "rag.answer_temperature", "rag.answer_max_tokens", "rag.evidence_span_enabled", "rag.evidence_span_window_sentences", "rag.evidence_span_max_chars", "rag.deterministic_tables_enabled", "rag.deterministic_table_max_rows", "rag.claim_verification_enabled", "rag.claim_verification_min_overlap"],
    "Runtime and LAN safety": ["runtime.max_chat_concurrency", "runtime.max_search_concurrency", "runtime.max_admin_jobs", "runtime.reject_chat_during_rebuild", "security.approved_lan_hosts"],
}


def override_path_for_config(config_path: str | Path) -> Path:
    return Path(config_path).with_name("ui_overrides.yaml")


def read_overrides(config_path: str | Path) -> dict[str, Any]:
    path = override_path_for_config(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"UI overrides must be a mapping: {path}")
    return data


def write_overrides(config_path: str | Path, overrides: dict[str, Any]) -> Path:
    path = override_path_for_config(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(overrides, file, sort_keys=False, allow_unicode=True)
    return path


def clear_overrides(config_path: str | Path) -> dict[str, Any]:
    path = override_path_for_config(config_path)
    if path.exists():
        path.unlink()
    return {"status": "ok", "message": "UI configuration overrides cleared.", "overrides": {}}


def get_nested(data: Any, dotted_path: str) -> Any:
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, Settings):
            current = getattr(current, part)
        elif hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def set_nested(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def validate_value(spec: ConfigFieldSpec, value: Any) -> Any:
    if spec.kind == "int":
        if isinstance(value, bool):
            raise ValueError(f"{spec.path} must be an integer.")
        value = int(value)
    elif spec.kind == "float":
        if isinstance(value, bool):
            raise ValueError(f"{spec.path} must be a number.")
        value = float(value)
    elif spec.kind == "bool":
        if isinstance(value, str):
            value = value.strip().lower() in {"1", "true", "yes", "on"}
        else:
            value = bool(value)
    elif spec.kind == "select":
        value = str(value).strip()
        if value not in spec.choices:
            raise ValueError(f"{spec.path} must be one of: {', '.join(spec.choices)}")
    elif spec.kind == "csv":
        if isinstance(value, list):
            value = [str(item).strip() for item in value if str(item).strip()]
        else:
            value = [item.strip() for item in str(value or "").split(",") if item.strip()]
    elif spec.kind == "text":
        value = str(value or "").strip()
        if spec.path not in {"ocr.tesseract_cmd", "security.approved_lan_hosts"} and not value:
            raise ValueError(f"{spec.path} cannot be empty.")
    else:
        raise ValueError(f"Unsupported config field type: {spec.kind}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if spec.minimum is not None and value < spec.minimum:
            raise ValueError(f"{spec.path} must be >= {spec.minimum}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ValueError(f"{spec.path} must be <= {spec.maximum}.")

    return value


def build_config_schema(settings: Settings) -> dict[str, Any]:
    return {
        "editable_groups": {
            group: [SAFE_FIELDS[path].to_dict(get_nested(settings, path)) for path in paths]
            for group, paths in GROUPS.items()
        },
        "local_only": {
            "allow_external_calls": settings.security.allow_external_calls,
            "approved_lan_hosts": settings.security.approved_lan_hosts,
            "note": "When allow_external_calls=false, model endpoints must be localhost, private LAN IPs, .local names, or explicitly approved LAN hostnames.",
        },
    }


def _validate_candidate(config_path: str | Path, candidate_overrides: dict[str, Any]) -> Settings:
    """Validate candidate overrides before touching ui_overrides.yaml."""

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(candidate_overrides, handle, sort_keys=False, allow_unicode=True)
        tmp_path = Path(handle.name)
    try:
        return load_settings(config_path, overrides_path=tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def apply_safe_overrides(config_path: str | Path, patch: dict[str, Any]) -> dict[str, Any]:
    flattened = flatten_patch(patch)
    current = read_overrides(config_path)

    # Validate scalar values first, then validate the full merged settings before writing.
    for path, value in flattened.items():
        spec = SAFE_FIELDS.get(path)
        if spec is None:
            raise ValueError(f"Field is not editable through UI: {path}")
        normalized = validate_value(spec, value)
        set_nested(current, path, normalized)

    settings = _validate_candidate(config_path, current)

    # Extra pre-write clarity for local URLs. The Settings validator also blocks these.
    for path, spec in SAFE_FIELDS.items():
        if not spec.local_url:
            continue
        value = get_nested(settings, path)
        if value:
            try:
                assert_allowed_endpoint(str(value), settings)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc

    write_overrides(config_path, current)
    return current


def flatten_patch(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_patch(value, path))
        else:
            flattened[path] = value
    return flattened
