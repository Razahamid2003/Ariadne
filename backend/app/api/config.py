"""Configuration endpoints for the settings drawer.

Purpose
-------
Lets the UI read the current effective settings and safely change a curated set
of tuning values without hand-editing YAML.

What it does
------------
Returns the effective configuration, the editable field schema for the settings
panel, and endpoints to validate-and-apply overrides, reload config from disk, or
clear overrides back to the baseline.

Flow
----
Updates are validated against the field schema, written to the overrides file,
and applied to the running app so changes take effect immediately. Only
whitelisted fields are editable; everything else stays read-only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.api.deps import get_rags_state
from backend.app.runtime.app_state import RAGSAppState
from backend.app.services.config_overrides import (
    apply_safe_overrides,
    build_config_schema,
    clear_overrides,
    read_overrides,
)

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigPatchRequest(BaseModel):
    """Patch body for safe UI overrides."""

    overrides: dict[str, Any] = Field(default_factory=dict)


@router.get("/effective")
def api_effective_config(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Return UI-safe effective config details."""

    return {
        "config_path": str(state.config_path),
        "overrides": read_overrides(state.config_path),
        "schema": build_config_schema(state.settings),
    }


@router.get("/schema")
def api_config_schema(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Return editable config schema for the settings drawer."""

    return build_config_schema(state.settings)


@router.patch("/overrides")
def api_update_overrides(payload: ConfigPatchRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Validate, save, and apply UI-safe config overrides."""

    try:
        overrides = apply_safe_overrides(state.config_path, payload.overrides)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.reload_settings()
    return {
        "status": "ok",
        "message": "Configuration overrides saved and runtime settings reloaded.",
        "overrides": overrides,
        "schema": build_config_schema(state.settings),
    }


@router.post("/reload")
def api_reload_config(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Reload client.yaml + ui_overrides.yaml."""

    state.reload_settings()
    return {
        "status": "ok",
        "message": "Runtime settings reloaded.",
        "schema": build_config_schema(state.settings),
    }


@router.delete("/overrides")
def api_clear_overrides(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Clear UI-saved configuration overrides and reload baseline config."""

    result = clear_overrides(state.config_path)
    state.reload_settings()
    result["schema"] = build_config_schema(state.settings)
    return result
