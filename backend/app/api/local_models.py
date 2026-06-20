"""Local model discovery and switching endpoints.

Purpose
-------
Lets the UI see which local models are installed and switch the active model,
without the browser ever talking to a model server directly.

What it does
------------
Detects models from local OpenAI-compatible and Ollama endpoints, reports the
current model status, and applies a chosen model by saving it to the overrides
file and reloading services.

Flow
----
All model-server traffic is mediated by the backend. Detection probes the
configured local endpoints, normalizes their model lists, and suggests a sensible
timeout for large models. Applying a model validates it, persists the choice, and
refreshes the runtime.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import get_rags_state
from backend.app.runtime.app_state import RAGSAppState
from backend.app.services.config_overrides import read_overrides, set_nested, write_overrides
from backend.app.services.network_safety import assert_allowed_endpoint

router = APIRouter(prefix="/api/models", tags=["local models"])


class DetectModelsRequest(BaseModel):
    """Optional local endpoints to inspect. Defaults come from active config."""

    llm_base_url: str | None = None
    vision_base_url: str | None = None


class ApplyModelsRequest(BaseModel):
    """Model override request saved to config/ui_overrides.yaml."""

    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: int | None = Field(default=None, ge=15, le=900)
    llm_max_tokens: int | None = Field(default=None, ge=64, le=4096)
    vision_base_url: str | None = None
    vision_model: str | None = None
    vision_timeout_seconds: int | None = Field(default=None, ge=15, le=900)


async def _get_json(url: str, timeout: float = 6.0) -> tuple[dict[str, Any] | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json(), None
    except Exception as exc:
        return None, str(exc)


def _openai_models_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        return f"{cleaned}/models"
    return f"{cleaned}/v1/models"


def _ollama_tags_url(base_url: str) -> str:
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return f"{cleaned}/api/tags"


def _extract_openai_models(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
    return sorted(dict.fromkeys(out))


def _extract_ollama_models(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    out = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
    return sorted(dict.fromkeys(out))


async def _detect_endpoint(base_url: str) -> dict[str, Any]:
    """Try OpenAI-compatible and Ollama-native model discovery locally."""

    openai_payload, openai_error = await _get_json(_openai_models_url(base_url))
    openai_models = _extract_openai_models(openai_payload)

    ollama_payload, ollama_error = await _get_json(_ollama_tags_url(base_url))
    ollama_models = _extract_ollama_models(ollama_payload)

    models = sorted(dict.fromkeys([*openai_models, *ollama_models]))
    return {
        "base_url": base_url,
        "models": models,
        "openai_compatible": {
            "url": _openai_models_url(base_url),
            "ok": bool(openai_models),
            "models": openai_models,
            "error": None if openai_models else openai_error,
        },
        "ollama_native": {
            "url": _ollama_tags_url(base_url),
            "ok": bool(ollama_models),
            "models": ollama_models,
            "error": None if ollama_models else ollama_error,
        },
    }


def _suggest_timeout(model_name: str, current_timeout: int | None) -> int:
    """Give large local models enough time on workstation GPUs.

    A 30B-class model can easily exceed 120 seconds for long RAG prompts on a
    laptop-class GPU, especially in detailed mode. This does not make it faster;
    it prevents the request from being cut off prematurely.
    """

    model = (model_name or "").lower()
    current = int(current_timeout or 120)
    if any(marker in model for marker in ("gemma4", "gemma-4", "31b", "30b", "32b", "33b", "34b", "70b", "72b")):
        return max(current, 420)
    if any(marker in model for marker in ("gemma3", "gemma", "14b", "13b", "12b")):
        return max(current, 240)
    return current


@router.get("/status")
def api_model_status(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    settings = state.settings
    return {
        "llm": {
            "provider": settings.llm.provider,
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
            "timeout_seconds": settings.llm.timeout_seconds,
            "max_tokens": settings.llm.max_tokens,
        },
        "vision": {
            "enabled": settings.vision.enabled,
            "provider": settings.vision.provider,
            "base_url": settings.vision.base_url,
            "model": settings.vision.model,
            "timeout_seconds": settings.vision.timeout_seconds,
        },
        "note": "The browser talks only to FastAPI. Model detection and switching are handled locally by the backend.",
    }


@router.post("/detect")
async def api_detect_models(payload: DetectModelsRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    llm_base = payload.llm_base_url or state.settings.llm.base_url
    vision_base = payload.vision_base_url or state.settings.vision.base_url
    try:
        assert_allowed_endpoint(llm_base, state.settings)
        assert_allowed_endpoint(vision_base, state.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "llm": await _detect_endpoint(llm_base),
        "vision": await _detect_endpoint(vision_base),
    }


@router.post("/apply")
def api_apply_models(payload: ApplyModelsRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    overrides = read_overrides(state.config_path)

    model_name = payload.llm_model.strip() if payload.llm_model is not None else state.settings.llm.model
    timeout = payload.llm_timeout_seconds
    if timeout is None and payload.llm_model is not None:
        timeout = _suggest_timeout(model_name, state.settings.llm.timeout_seconds)

    try:
        if payload.llm_base_url is not None:
            assert_allowed_endpoint(payload.llm_base_url, state.settings)
        if payload.vision_base_url is not None:
            assert_allowed_endpoint(payload.vision_base_url, state.settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.llm_base_url is not None:
        set_nested(overrides, "llm.base_url", payload.llm_base_url.strip())
    if payload.llm_model is not None:
        set_nested(overrides, "llm.model", model_name)
    if timeout is not None:
        set_nested(overrides, "llm.timeout_seconds", int(timeout))
    if payload.llm_max_tokens is not None:
        set_nested(overrides, "llm.max_tokens", int(payload.llm_max_tokens))
    if payload.vision_base_url is not None:
        set_nested(overrides, "vision.base_url", payload.vision_base_url.strip())
    if payload.vision_model is not None:
        set_nested(overrides, "vision.model", payload.vision_model.strip())
    if payload.vision_timeout_seconds is not None:
        set_nested(overrides, "vision.timeout_seconds", int(payload.vision_timeout_seconds))

    try:
        write_overrides(state.config_path, overrides)
        state.reload_settings()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "message": "Local model settings were saved and the runtime was reloaded.",
        "llm": {
            "base_url": state.settings.llm.base_url,
            "model": state.settings.llm.model,
            "timeout_seconds": state.settings.llm.timeout_seconds,
            "max_tokens": state.settings.llm.max_tokens,
        },
        "vision": {
            "base_url": state.settings.vision.base_url,
            "model": state.settings.vision.model,
            "timeout_seconds": state.settings.vision.timeout_seconds,
        },
    }
