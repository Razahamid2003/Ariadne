"""System-status endpoint.

Purpose
-------
Returns a browser-friendly snapshot of runtime and index state for the readiness
panel: document counts, whether the keyword and vector indexes are built, and the
active model.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_rags_state
from backend.app.runtime.app_state import RAGSAppState
from backend.app.services.system_status import build_system_status

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
def api_status(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    payload = build_system_status(state.settings, config_path=str(state.config_path))
    payload["runtime_state"] = state.summary()
    payload["jobs"] = {
        "active": [job.to_dict(include_result=False) for job in state.job_manager.active_jobs()],
        "recent": [job.to_dict(include_result=False) for job in state.job_manager.list(limit=5)],
    }
    return payload
