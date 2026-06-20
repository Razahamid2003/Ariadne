"""Evidence-search endpoint.

Purpose
-------
Runs hybrid retrieval directly and returns the ranked evidence without generating
an answer, used by the "source trail" search in the UI.

Flow
----
Takes a query, calls the hybrid retriever, and returns the ranked candidates with
their scores and source metadata.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_rags_state
from backend.app.api.models import SearchRequest
from backend.app.retrieval.models import HybridSearchRequest
from backend.app.runtime.app_state import RAGSAppState
from backend.app.rag.diagnostics import single_pass_diagnostics

router = APIRouter(prefix="/api", tags=["search"])

INDEX_MUTATING_JOBS = {"ingest", "rebuild", "fresh_rebuild"}


@router.post("/search")
async def api_search(payload: SearchRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    if state.index_mutating() or state.job_manager.has_active_job(INDEX_MUTATING_JOBS):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ariadne is refreshing the source trail. Try again when the rebuild finishes.",
        )

    started = time.perf_counter()

    async with state.concurrency.search_slot():
        retriever = state.get_retriever()
        response = retriever.search(
            HybridSearchRequest(
                query=payload.query,
                top_k=payload.top_k,
                source_system=payload.source_system,
                record_type=payload.record_type,
            )
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    result = response.to_dict(preview_chars=payload.preview_chars)
    result["diagnostics"] = single_pass_diagnostics(result.get("diagnostics") or {})
    result["latency_ms"] = latency_ms

    state.query_logger.log(
        {
            "type": "search",
            "query": payload.query,
            "status": "ok" if response.results else "empty",
            "confidence": response.confidence,
            "latency_ms": latency_ms,
            "result_count": len(response.results),
        }
    )

    return result
