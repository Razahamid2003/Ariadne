"""Local/LAN endpoint safety checks.

Purpose
-------
Ensures model and vision endpoints point only at localhost or the local network
when external calls are disabled, preventing accidental cloud calls.

What it does
------------
Normalizes hostnames and validates that configured URLs resolve to loopback,
private LAN ranges, ``.local`` names, or explicitly approved hosts. A helper
validates a whole set of endpoints under the same policy.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

LOCAL_HOSTNAMES = {"localhost"}
LOCAL_IPS = {
    ipaddress.ip_address("127.0.0.1"),
    ipaddress.ip_address("::1"),
}


def normalize_host(host: str | None) -> str:
    """Normalize a URL hostname for allow-list checks."""

    return (host or "").strip().lower().strip("[]")


def is_private_or_local_host(host: str, approved_hosts: list[str] | tuple[str, ...] | None = None) -> bool:
    """Return True when host is localhost, private LAN IP, .local, or approved.

    This function intentionally does not resolve arbitrary hostnames. Resolving
    public-looking names can create DNS traffic outside the LAN, which is exactly
    what the local/offline mode is trying to avoid.
    """

    normalized = normalize_host(host)
    if not normalized:
        return False

    approved = {normalize_host(item) for item in (approved_hosts or []) if item}
    if normalized in approved:
        return True

    if normalized in LOCAL_HOSTNAMES:
        return True

    if normalized.endswith(".local"):
        return True

    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False

    return bool(
        ip in LOCAL_IPS
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
    )


def validate_local_url(
    url: str,
    *,
    allow_external_calls: bool = False,
    approved_hosts: list[str] | tuple[str, ...] | None = None,
    field_name: str = "endpoint",
) -> str:
    """Validate that a URL points to localhost/LAN unless external calls are allowed."""

    value = (url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an http(s) URL with a host: {url!r}")

    if allow_external_calls:
        return value

    if not is_private_or_local_host(parsed.hostname, approved_hosts=approved_hosts):
        raise ValueError(
            f"{field_name} is blocked because security.allow_external_calls=false. "
            "Use localhost, 127.0.0.1, a private LAN IP such as 192.168.x.x/10.x.x.x, "
            "a .local hostname, or add an explicit approved LAN hostname. "
            f"Blocked host: {parsed.hostname}"
        )

    return value


def validate_local_endpoint_set(
    endpoints: dict[str, str],
    *,
    allow_external_calls: bool,
    approved_hosts: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Validate several endpoint URLs using the same local-only policy."""

    for field_name, url in endpoints.items():
        validate_local_url(
            url,
            allow_external_calls=allow_external_calls,
            approved_hosts=approved_hosts,
            field_name=field_name,
        )
