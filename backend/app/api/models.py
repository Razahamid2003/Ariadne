"""Request body schemas for the API.

Purpose
-------
Defines the typed request models (search, chat, create-chat, ingest, rebuild)
that FastAPI uses to validate and parse incoming JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)
    source_system: str | None = None
    record_type: str | None = None
    preview_chars: int = Field(default=700, ge=100, le=5000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    chat_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    source_system: str | None = None
    record_type: str | None = None
    show_evidence: bool = True
    answer_mode: Literal["brief", "balanced", "detailed"] = "balanced"
    preview_chars: int = Field(default=700, ge=100, le=5000)


class CreateChatRequest(BaseModel):
    title: str | None = None


class IngestRequest(BaseModel):
    force: bool = False


class RebuildRequest(BaseModel):
    fresh: bool = False
    extract: bool = False
    clear_extract: bool = False
