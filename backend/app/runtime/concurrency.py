"""Concurrency controls.

Purpose
-------
Limits how many expensive operations run at once so a single machine serving
several local users is not overwhelmed.

What it does
------------
Provides async gates (semaphores) for chat and search work and reports a snapshot
of current usage for status endpoints.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(frozen=True)
class ConcurrencySnapshot:
    """Human-readable concurrency state for status/debug endpoints."""

    max_chat_concurrency: int
    max_search_concurrency: int
    max_admin_jobs: int
    reject_chat_during_rebuild: bool

    def to_dict(self) -> dict:
        return {
            "max_chat_concurrency": self.max_chat_concurrency,
            "max_search_concurrency": self.max_search_concurrency,
            "max_admin_jobs": self.max_admin_jobs,
            "reject_chat_during_rebuild": self.reject_chat_during_rebuild,
        }


class ConcurrencyManager:
    """
    Lightweight async concurrency gate for API requests.

    FastAPI can accept many requests concurrently. The local model stack cannot
    always process all of them at once, especially on a laptop GPU. These gates
    let the server queue chat/search requests instead of duplicating model loads
    or saturating the machine.
    """

    def __init__(self, settings):
        runtime = settings.runtime
        self._chat_semaphore = asyncio.Semaphore(runtime.max_chat_concurrency)
        self._search_semaphore = asyncio.Semaphore(runtime.max_search_concurrency)
        self.max_chat_concurrency = runtime.max_chat_concurrency
        self.max_search_concurrency = runtime.max_search_concurrency
        self.max_admin_jobs = runtime.max_admin_jobs
        self.reject_chat_during_rebuild = runtime.reject_chat_during_rebuild

    @asynccontextmanager
    async def chat_slot(self) -> AsyncIterator[None]:
        async with self._chat_semaphore:
            yield

    @asynccontextmanager
    async def search_slot(self) -> AsyncIterator[None]:
        async with self._search_semaphore:
            yield

    def snapshot(self) -> ConcurrencySnapshot:
        return ConcurrencySnapshot(
            max_chat_concurrency=self.max_chat_concurrency,
            max_search_concurrency=self.max_search_concurrency,
            max_admin_jobs=self.max_admin_jobs,
            reject_chat_during_rebuild=self.reject_chat_during_rebuild,
        )
