"""Local language-model interface.

Purpose
-------
Defines the common contract every language-model client implements, so the rest of
the system depends on an interface rather than a specific provider.

What it does
------------
``LLMResponse`` is the standard result shape; ``LLMClient`` is the protocol with a
single ``generate()`` method that all clients follow.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    """
    Standard response returned by all LLM clients.

    Attributes:
        text:
            Generated model output. Empty if status is "error".

        model:
            Configured model name used for the request.

        status:
            "ok" or "error".

        error:
            Error message if the call failed.

        latency_ms:
            Request duration in milliseconds.
    """

    text: str
    model: str
    status: str
    error: str | None = None
    latency_ms: int | None = None


class LLMClient(Protocol):
    """
    Protocol that all LLM clients must follow.
    """

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Generate a response from a local LLM.

        Args:
            system_prompt:
                Instruction/developer context for the model.

            user_prompt:
                User message or task.

        Returns:
            LLMResponse:
                Normalized response object.
        """
        ...