"""Endpoint boundary checks for services.

Purpose
-------
Guards model-service endpoints at the service layer, refusing any endpoint outside
the configured local/LAN boundary.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from backend.app.core.config import Settings


LOCAL_NAMES = {"localhost"}


def assert_allowed_endpoint(url: str, settings: Settings) -> None:
    """Raise ValueError when an endpoint is outside the configured local boundary."""

    if getattr(settings.security, "allow_external_calls", False):
        return

    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Model endpoint must be an HTTP URL with a host.")

    host = parsed.hostname.lower().strip("[]")
    if _is_local_or_lan_host(host, settings):
        return

    raise ValueError(
        "Model endpoint must stay on this machine or an approved LAN host while external calls are disabled."
    )


def _is_local_or_lan_host(host: str, settings: Settings) -> bool:
    if host in LOCAL_NAMES or host.endswith(".localhost"):
        return True
    if host.endswith(".local"):
        return True
    approved = {str(item).lower().strip() for item in getattr(settings.security, "approved_lan_hosts", []) or []}
    if host in approved:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
    except ValueError:
        return False
