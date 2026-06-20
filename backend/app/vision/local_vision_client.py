"""Optional local vision client.

Purpose
-------
Generates captions for images using a local vision-capable model, only when vision
is explicitly enabled.

What it does
------------
Sends an image to the configured local vision endpoint and returns a caption,
extracting the text from the model's reply.

Flow
----
When enabled, an image (file or bytes) is sent to the local vision model and the
caption is returned for inclusion in the index; when disabled, this path is skipped.
"""

import base64
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.app.core.config import VisionConfig


@dataclass(frozen=True)
class VisionResult:
    """
    Result from optional local vision captioning.
    """

    status: str
    caption: str
    error: str | None = None


class LocalVisionClient:
    """
    Optional local vision client.
    """

    def __init__(self, config: VisionConfig):
        self.config = config

    def is_enabled(self) -> bool:
        """
        Return True only when captioning is explicitly enabled.
        """

        return self.config.enabled and self.config.mode == "caption"

    def caption_image_file(
        self,
        image_path: str | Path,
        prompt: str | None = None,
    ) -> VisionResult:
        """
        Caption an image file.
        """

        path = Path(image_path)

        if not path.exists():
            return VisionResult(
                status="error",
                caption="",
                error=f"Image not found: {path}",
            )

        try:
            image_bytes = path.read_bytes()
        except Exception as exc:
            return VisionResult(
                status="error",
                caption="",
                error=f"Could not read image {path}: {exc}",
            )

        return self.caption_image_bytes(
            image_bytes=image_bytes,
            image_label=path.name,
            prompt=prompt,
        )

    def caption_image_bytes(
        self,
        image_bytes: bytes,
        image_label: str,
        prompt: str | None = None,
    ) -> VisionResult:
        """
        Caption image bytes using the configured local vision provider.
        """

        if not self.is_enabled():
            return VisionResult(
                status="disabled",
                caption="",
                error="Vision captioning is disabled.",
            )

        if self.config.provider == "ollama":
            return self._caption_with_ollama_chat(
                image_bytes=image_bytes,
                image_label=image_label,
                prompt=prompt,
            )

        return VisionResult(
            status="error",
            caption="",
            error=f"Unsupported vision provider: {self.config.provider}",
        )

    def _caption_with_ollama_chat(
        self,
        image_bytes: bytes,
        image_label: str,
        prompt: str | None = None,
    ) -> VisionResult:
        """
        Caption image bytes using Ollama /api/chat.
        """

        api_url = f"{self.config.base_url.rstrip('/')}/api/chat"
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        final_prompt = prompt or (
            "Describe this image for a retrieval system. "
            "Focus on visible objects, equipment, labels, readable text, logos, "
            "model numbers, product names, company names, and useful identifying details. "
            "If text is not readable, say so. Do not invent details."
        )

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": final_prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(api_url, json=payload)

                if response.status_code >= 400:
                    return VisionResult(
                        status="error",
                        caption="",
                        error=(
                            f"Ollama vision request failed with HTTP {response.status_code}.\n"
                            f"URL: {api_url}\n"
                            f"Response body:\n{response.text[:2000]}"
                        ),
                    )

                data = response.json()

            caption = self._extract_ollama_chat_text(data)

            if not caption:
                return VisionResult(
                    status="error",
                    caption="",
                    error=f"Vision model returned an empty caption for {image_label}. Response: {data}",
                )

            return VisionResult(
                status="ok",
                caption=caption,
                error=None,
            )

        except Exception as exc:
            return VisionResult(
                status="error",
                caption="",
                error=str(exc),
            )

    @staticmethod
    def _extract_ollama_chat_text(data: dict) -> str:
        """
        Extract text from Ollama /api/chat response.
        """

        message = data.get("message")

        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return str(content).strip()

        response = data.get("response")

        if response:
            return str(response).strip()

        return ""