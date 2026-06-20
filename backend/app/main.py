"""Application entry point.

Purpose
-------
Creates and configures the FastAPI application: it wires up the API routes, serves
the browser UI from bundled static files, and applies the offline posture on
startup.

What it does
------------
Registers all routers, serves the main and admin pages and the favicon, exposes
health endpoints, and runs startup/shutdown logic. All UI assets are local; there
are no cloud or CDN dependencies.

Flow
----
On startup the lifespan handler loads settings, applies the air-gap hardening, and
prepares shared services; routes are then served until shutdown.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core import airgap
airgap.apply_offline_env()  # offline/no-telemetry flags before ML libs load

from backend.app.api.admin import router as admin_router
from backend.app.api.chat import router as chat_router
from backend.app.api.config import router as config_router
from backend.app.api.local_models import router as local_models_router
from backend.app.api.search import router as search_router
from backend.app.api.status import router as status_router
from backend.app.core.config import load_settings
from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient
from backend.app.runtime.app_state import RAGSAppState

CONFIG_PATH = os.environ.get("RAGS_CONFIG_PATH", "config/client.yaml")
UI_DIR = Path(__file__).resolve().parent / "ui" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rags = RAGSAppState(CONFIG_PATH)
    airgap.harden(app.state.rags.settings)
    try:
        yield
    finally:
        app.state.rags.job_manager.shutdown()


_startup_settings = load_settings(CONFIG_PATH)
_airgap_status = airgap.harden(_startup_settings)
app = FastAPI(title=_startup_settings.app.name, lifespan=lifespan)

app.include_router(status_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(config_router)
app.include_router(local_models_router)

if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui_index():
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return {
        "status": "ok",
        "app": _startup_settings.app.name,
        "message": "RAGS PoC API is running locally. UI files are not installed.",
        "docs": "/docs",
        "status_endpoint": "/api/status",
    }


@app.get("/admin", include_in_schema=False)
def ui_admin():
    admin_path = UI_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path)
    return FileResponse(UI_DIR / "index.html") if (UI_DIR / "index.html").exists() else {"status": "missing_ui"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = UI_DIR / "assets" / "favicon.svg"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/health")
def health() -> dict:
    settings = getattr(app.state, "rags", None).settings if hasattr(app.state, "rags") else _startup_settings
    return {
        "status": "ok",
        "app": settings.app.name,
        "offline_mode": settings.app.offline_mode,
        "deployment_mode": settings.deployment.mode,
        "airgap_egress_guard": airgap._GUARD_INSTALLED,
        "allow_external_calls": settings.security.allow_external_calls,
    }


@app.get("/health/llm")
async def health_llm() -> dict:
    settings = getattr(app.state, "rags", None).settings if hasattr(app.state, "rags") else _startup_settings
    llm = OpenAICompatibleLLMClient(settings.llm)

    response = await llm.generate(
        system_prompt="You are a health check assistant. Reply briefly.",
        user_prompt="Reply with: ok",
    )

    return {
        "status": response.status,
        "model": response.model,
        "latency_ms": response.latency_ms,
        "error": response.error,
    }
