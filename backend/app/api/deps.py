"""Shared FastAPI dependencies.

Purpose
-------
Provides the dependency that hands request handlers the shared application state
(retriever, answer generator, stores) attached to the running server.
"""

from __future__ import annotations

from fastapi import Request

from backend.app.runtime.app_state import RAGSAppState


def get_rags_state(request: Request) -> RAGSAppState:
    return request.app.state.rags
