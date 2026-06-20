"""Minimal HTTP client for the local API.

Purpose
-------
A tiny standard-library HTTP client used by the evaluation tools to call the
running Ariadne API, avoiding any extra networking dependency.

What it does
------------
Posts JSON to the chat and search endpoints and returns the parsed result.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    status_code: int | None
    payload: dict[str, Any]
    latency_ms: int
    error: str | None = None


class AriadneApiClient:
    """HTTP client that avoids adding a requests/httpx dependency."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post_json(self, path: str, body: dict[str, Any]) -> ApiResult:
        url = f"{self.base_url}/{path.lstrip('/')}"
        data = json.dumps(body).encode("utf-8")
        request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - local operator-supplied URL
                raw = response.read().decode("utf-8")
                latency_ms = int((time.perf_counter() - started) * 1000)
                return ApiResult(True, response.status, json.loads(raw or "{}"), latency_ms)
        except HTTPError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                raw = exc.read().decode("utf-8")
                payload = json.loads(raw or "{}")
            except Exception:
                payload = {}
            return ApiResult(False, exc.code, payload, latency_ms, error=f"HTTP {exc.code}: {payload or exc.reason}")
        except URLError as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ApiResult(False, None, {}, latency_ms, error=f"Connection error: {exc.reason}")
        except Exception as exc:  # pragma: no cover - defensive CLI path
            latency_ms = int((time.perf_counter() - started) * 1000)
            return ApiResult(False, None, {}, latency_ms, error=str(exc))

    def chat(self, query: str, **kwargs: Any) -> ApiResult:
        body = {"query": query, **kwargs}
        return self.post_json("/api/chat", body)

    def search(self, query: str, **kwargs: Any) -> ApiResult:
        body = {"query": query, **kwargs}
        return self.post_json("/api/search", body)
