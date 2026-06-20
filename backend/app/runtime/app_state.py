"""Shared application state.

Purpose
-------
Keeps the heavy, shared services (retriever, answer generator, stores) in one place
for the whole server process, so requests reuse them instead of recreating them.

What it does
------------
Holds the loaded settings and lazily-built services, validates local endpoints,
exposes the retriever and answer generator, and coordinates safe index mutations.

Flow
----
The server attaches one instance to the app at startup; handlers fetch services
from it. When the index is being updated, mutation guards prevent serving from a
half-updated index, and services are reset after a config reload.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock
import gc

from backend.app.core.config import load_settings
from backend.app.jobs.job_manager import JobManager
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.runtime.concurrency import ConcurrencyManager
from backend.app.services.query_logger import QueryLogger
from backend.app.services.chat_store import ChatStore
from backend.app.services.network_safety import assert_allowed_endpoint


class RAGSAppState:
    """
    Shared app state attached to FastAPI's app.state.rags.

    Retrieval and RAG services are lazy-loaded once. This keeps startup fast and
    still prevents per-request model/index reloads. Admin rebuild jobs can reset
    the services after indexes are refreshed.
    """

    def __init__(self, config_path: str | Path = "config/client.yaml"):
        self.config_path = Path(config_path)
        self.settings = load_settings(self.config_path)
        self._validate_local_endpoints()
        self.project_root = Path.cwd()
        self.concurrency = ConcurrencyManager(self.settings)
        self.query_logger = QueryLogger(self.settings.paths.logs)
        self.chat_store = ChatStore(self.settings.paths.logs)
        self.job_manager = JobManager(max_workers=self.settings.runtime.max_admin_jobs)
        self._lock = RLock()
        self._retriever: HybridRetriever | None = None
        self._rag_answer_generator: RAGAnswerGenerator | None = None
        self._index_mutating = False

    def reload_settings(self) -> None:
        """Reload config from disk and reset runtime-managed services."""

        with self._lock:
            self.settings = load_settings(self.config_path)
            self._validate_local_endpoints()
            self.concurrency = ConcurrencyManager(self.settings)
            self.query_logger = QueryLogger(self.settings.paths.logs)
            self.chat_store = ChatStore(self.settings.paths.logs)
            self.job_manager.resize(self.settings.runtime.max_admin_jobs)
            self._rag_answer_generator = None
            self._retriever = None
            gc.collect()


    def _validate_local_endpoints(self) -> None:
        """Keep configured model endpoints within the local/LAN boundary."""

        assert_allowed_endpoint(self.settings.llm.base_url, self.settings)
        assert_allowed_endpoint(self.settings.vision.base_url, self.settings)

    def get_retriever(self) -> HybridRetriever:
        """Return the shared HybridRetriever, creating it once if needed."""

        with self._lock:
            if self._retriever is None:
                self._retriever = HybridRetriever(self.settings)
            return self._retriever

    def get_rag_answer_generator(self) -> RAGAnswerGenerator:
        """Return the shared RAG answer generator, creating it once if needed."""

        with self._lock:
            if self._rag_answer_generator is None:
                self._rag_answer_generator = RAGAnswerGenerator(
                    self.settings,
                    retriever=self.get_retriever(),
                )
            return self._rag_answer_generator

    def reset_rag_services(self) -> None:
        """
        Drop cached retrieval/RAG services after index-mutating admin jobs.

        The next chat/search request will lazily rebuild service objects against
        the latest SQLite/keyword/vector files.
        """

        with self._lock:
            self._rag_answer_generator = None
            self._retriever = None
            gc.collect()

    def begin_index_mutation(self) -> None:
        """Enter an index-changing maintenance window.

        Cached retrieval/RAG objects are released before fresh rebuilds so
        Windows can unlock SQLite/vector files before they are replaced.
        """

        with self._lock:
            self._index_mutating = True
            self._rag_answer_generator = None
            self._retriever = None
            gc.collect()

    def end_index_mutation(self) -> None:
        """Leave an index-changing maintenance window."""

        with self._lock:
            self._rag_answer_generator = None
            self._retriever = None
            self._index_mutating = False
            gc.collect()

    def index_mutating(self) -> bool:
        """Return True while ingestion/rebuild is replacing local indexes."""

        with self._lock:
            return self._index_mutating

    def summary(self) -> dict:
        return {
            "config_path": str(self.config_path),
            "services_loaded": {
                "retriever": self._retriever is not None,
                "rag_answer_generator": self._rag_answer_generator is not None,
            },
            "index_mutating": self.index_mutating(),
            "concurrency": self.concurrency.snapshot().to_dict(),
        }
