"""Health-check endpoints.

Purpose
-------
Lightweight endpoints that confirm the server is up and the local language model
is reachable, used for readiness checks and monitoring.

Flow
----
The router is built against the active settings and reports basic liveness plus
the configured local model endpoint's reachability.
"""

from fastapi import APIRouter

from backend.app.core.config import Settings
from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient

router = APIRouter(tags=["health"])


def build_health_router(settings: Settings) -> APIRouter:
    """
    Build a health router bound to the current application settings.

    Keeping settings injection here avoids global config access inside route
    handlers and makes the app easier to test later.
    """

    @router.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "app": settings.app.name,
            "offline_mode": settings.app.offline_mode,
        }

    @router.get("/health/llm")
    async def health_llm() -> dict:
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

    return router