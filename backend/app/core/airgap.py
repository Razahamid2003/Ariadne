"""Air-gap enforcement for fully offline operation.

Purpose
-------
Guarantees that no data leaves the machine or local network, even if a third-party
library tries to open its own network connection.

What it does
------------
Sets offline environment flags for common ML libraries and installs a guard that
intercepts outbound socket connections, allowing only loopback and local-network
targets and blocking everything else.

Flow
----
``harden()`` applies the full offline posture from settings: it sets the offline
env flags and installs the egress guard. The guard wraps socket connect/resolution
and raises if a non-local address is attempted. A pure decision function decides
allow/block so the policy can be unit-tested, and the guard can be uninstalled in
tests.
"""

from __future__ import annotations

import ipaddress
import os
import socket

from backend.app.core.network_safety import is_private_or_local_host

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "DO_NOT_TRACK": "1",
    "SENTENCE_TRANSFORMERS_HOME": os.environ.get("SENTENCE_TRANSFORMERS_HOME", ""),
}

_GUARD_INSTALLED = False
_ORIG_CONNECT = None
_ORIG_CONNECT_EX = None
_ORIG_GETADDRINFO = None


def apply_offline_env() -> None:
    """Set offline/no-telemetry environment flags (idempotent)."""

    for key, value in _OFFLINE_ENV.items():
        if value == "" and key not in os.environ:
            continue
        os.environ.setdefault(key, value)


def _host_is_allowed(host: str | None, approved: tuple[str, ...]) -> bool:
    """Allow loopback, private/link-local IPs, .local names, localhost, the
    machine's own hostname, and explicitly approved LAN hosts. Block all else.
    """

    if host is None:
        return True  # AF_UNIX / None target
    if isinstance(host, (bytes, bytearray)):
        try:
            host = host.decode("ascii", "ignore")
        except Exception:
            return False
    normalized = str(host).strip().lower().strip("[]")
    if not normalized:
        return True
    if normalized in {"localhost", socket.gethostname().lower()}:
        return True
    if is_private_or_local_host(normalized, approved_hosts=list(approved)):
        return True
    try:
        ip = ipaddress.ip_address(normalized)
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
    except ValueError:
        # A non-numeric, non-approved hostname: block (prevents DNS egress too).
        return False


def should_block_address(address, approved: tuple[str, ...] = ()) -> bool:
    """Pure decision function (testable): True if this connect target is egress."""

    if not isinstance(address, tuple) or not address:
        return False  # AF_UNIX path string etc.
    host = address[0]
    return not _host_is_allowed(host, approved)


class EgressBlockedError(OSError):
    """Raised when an outbound connection to a non-local host is attempted."""


def install_egress_guard(approved_hosts: list[str] | tuple[str, ...] | None = None) -> bool:
    """Monkeypatch socket connect/resolution to block non-local egress.

    Returns True if installed, False if already installed. Idempotent.
    """

    global _GUARD_INSTALLED, _ORIG_CONNECT, _ORIG_CONNECT_EX, _ORIG_GETADDRINFO
    if _GUARD_INSTALLED:
        return False

    approved = tuple(approved_hosts or ())
    _ORIG_CONNECT = socket.socket.connect
    _ORIG_CONNECT_EX = socket.socket.connect_ex
    _ORIG_GETADDRINFO = socket.getaddrinfo

    def guarded_connect(self, address):
        if should_block_address(address, approved):
            raise EgressBlockedError(
                f"Air-gap: outbound connection to {address!r} blocked "
                "(security.allow_external_calls=false). Only localhost/LAN is permitted."
            )
        return _ORIG_CONNECT(self, address)

    def guarded_connect_ex(self, address):
        if should_block_address(address, approved):
            raise EgressBlockedError(
                f"Air-gap: outbound connection to {address!r} blocked "
                "(security.allow_external_calls=false)."
            )
        return _ORIG_CONNECT_EX(self, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _host_is_allowed(host, approved):
            raise EgressBlockedError(
                f"Air-gap: DNS resolution of {host!r} blocked "
                "(security.allow_external_calls=false)."
            )
        return _ORIG_GETADDRINFO(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    _GUARD_INSTALLED = True
    return True


def uninstall_egress_guard() -> None:
    """Restore original socket functions (used by tests)."""

    global _GUARD_INSTALLED, _ORIG_CONNECT, _ORIG_CONNECT_EX, _ORIG_GETADDRINFO
    if not _GUARD_INSTALLED:
        return
    if _ORIG_CONNECT:
        socket.socket.connect = _ORIG_CONNECT
    if _ORIG_CONNECT_EX:
        socket.socket.connect_ex = _ORIG_CONNECT_EX
    if _ORIG_GETADDRINFO:
        socket.getaddrinfo = _ORIG_GETADDRINFO
    _GUARD_INSTALLED = False


def harden(settings) -> dict:
    """Apply the full air-gap posture based on settings. Returns a status dict."""

    apply_offline_env()
    allow_external = bool(getattr(settings.security, "allow_external_calls", False))
    if allow_external:
        return {"offline_env": True, "egress_guard": False, "reason": "allow_external_calls=true"}
    approved = tuple(getattr(settings.security, "approved_lan_hosts", []) or [])
    installed = install_egress_guard(approved)
    return {"offline_env": True, "egress_guard": True, "newly_installed": installed, "approved_lan_hosts": list(approved)}


# Offline env must be set as early as possible, before ML libs are imported.
apply_offline_env()
