"""OpenAI-compatible local model client.

Purpose
-------
Talks to a local model server that speaks the OpenAI chat-completions format
(such as Ollama or other local runtimes).

What it does
------------
Sends a system/user prompt to the configured local endpoint, extracts the reply
text, and reports latency, returning the standard response shape.

Flow
----
``generate()`` builds the request, posts it to the local endpoint, parses the
completion, and returns text plus timing and status.
"""

import time
from typing import Any

import httpx

from backend.app.core.config import LLMConfig
from backend.app.core.logging import get_logger
from backend.app.llm.base import LLMResponse

logger = get_logger(__name__)


class OpenAICompatibleLLMClient:
    """
    Generic client for local OpenAI-compatible chat completion endpoints.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.chat_url = f"{config.base_url.rstrip('/')}/chat/completions"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Generate text from the configured local LLM.

        The method intentionally returns a normalized LLMResponse instead of
        raising model/API exceptions. This keeps API routes and future RAG
        orchestration code simpler and more predictable.
        """

        started = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(self.chat_url, json=payload)
                response.raise_for_status()
                data = response.json()

            text = self._extract_text(data)
            latency_ms = self._latency_ms(started)

            return LLMResponse(
                text=text,
                model=self.config.model,
                status="ok",
                error=None,
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = self._latency_ms(started)
            logger.warning("Local LLM call failed: %s", exc)

            return LLMResponse(
                text="",
                model=self.config.model,
                status="error",
                error=str(exc),
                latency_ms=latency_ms,
            )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """
        Extract generated text from an OpenAI-compatible response.

        Expected shape:
            {
              "choices": [
                {
                  "message": {
                    "content": "..."
                  }
                }
              ]
            }
        """

        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected LLM response format: {data}") from exc

    @staticmethod
    def _latency_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)