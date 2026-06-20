"""Admin endpoints for ingestion and index maintenance.

Purpose
-------
Lets an operator drive ingestion and index rebuilds from the browser instead of
the command line, and review or clear the local question log.

What it does
------------
Exposes endpoints to plan an ingest (preview what would change), run an
incremental ingest, rebuild the full index, list and inspect background jobs,
read or clear the local query log, and reload configuration.

Flow
----
Each endpoint validates its request, then hands the heavy work to the background
job manager so the browser never blocks. The handler returns a job handle the UI
can poll, or for quick actions returns the result directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_rags_state
from backend.app.api.models import IngestRequest, RebuildRequest
from backend.app.runtime.app_state import RAGSAppState
from backend.app.services.admin_operations import plan_ingest, run_incremental_ingest, run_rebuild

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _job_response(job) -> dict:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "status": job.status,
        "created_at": job.created_at,
    }


@router.post("/plan-ingest")
def api_plan_ingest(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    return plan_ingest(state.settings)


@router.post("/ingest")
def api_ingest(payload: IngestRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    if state.job_manager.has_active_job({"ingest", "rebuild", "fresh_rebuild"}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another ingestion or rebuild job is already running.",
        )

    def job_func() -> dict:
        state.begin_index_mutation()
        try:
            result = run_incremental_ingest(state.settings, force=payload.force)
            return result
        finally:
            state.end_index_mutation()

    job = state.job_manager.submit("ingest", job_func)
    return _job_response(job)


@router.post("/rebuild")
def api_rebuild(payload: RebuildRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    if state.job_manager.has_active_job({"ingest", "rebuild", "fresh_rebuild"}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another ingestion or rebuild job is already running.",
        )

    job_name = "fresh_rebuild" if payload.fresh else "rebuild"

    def job_func() -> dict:
        state.begin_index_mutation()
        try:
            result = run_rebuild(
                state.settings,
                fresh=payload.fresh,
                extract=payload.extract,
                clear_extract=payload.clear_extract,
            )
            return result
        finally:
            state.end_index_mutation()

    job = state.job_manager.submit(job_name, job_func)
    return _job_response(job)


@router.get("/jobs")
def api_jobs(limit: int = 20, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    return {
        "jobs": [job.to_dict(include_result=False) for job in state.job_manager.list(limit=limit)]
    }


@router.get("/jobs/{job_id}")
def api_job(job_id: str, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    job = state.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job.to_dict(include_result=True)


@router.get("/query-log")
def api_query_log(limit: int = 50, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    return {"records": state.query_logger.recent(limit=limit)}


@router.delete("/query-log")
def api_clear_query_log(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Clear local question history logs."""

    return state.query_logger.clear()


@router.post("/reload")
def api_reload(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    state.reload_settings()
    return {"status": "ok", "message": "Settings reloaded and runtime services reset."}
