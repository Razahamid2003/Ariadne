"""Tests for the air-gap egress guard.

Purpose
-------
Verifies that outbound connections to public hosts are blocked before any
connection is made, while loopback and approved local hosts are allowed through.
"""

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core import airgap
from backend.app.core.config import Settings


def test_decision_function():
    assert airgap.should_block_address(("8.8.8.8", 53)) is True
    assert airgap.should_block_address(("142.250.0.0", 443)) is True
    assert airgap.should_block_address(("huggingface.co", 443)) is True   # hostname -> blocked
    assert airgap.should_block_address(("127.0.0.1", 11434)) is False     # Ollama loopback
    assert airgap.should_block_address(("::1", 11434)) is False
    assert airgap.should_block_address(("192.168.1.50", 8080)) is False   # private LAN
    assert airgap.should_block_address(("10.0.0.5", 8080)) is False
    assert airgap.should_block_address(("localhost", 11434)) is False
    print("  block public IPs + hostnames; allow loopback/private/localhost  ✓")


def test_approved_lan_host_allowed():
    assert airgap.should_block_address(("models.lan", 9000), approved=("models.lan",)) is False
    assert airgap.should_block_address(("models.lan", 9000), approved=()) is True
    print("  approved LAN host allowlist works  ✓")


def test_offline_env_applied():
    airgap.apply_offline_env()
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"
    print("  offline env flags set (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / telemetry off)  ✓")


def test_guard_blocks_public_allows_loopback():
    airgap.install_egress_guard(approved_hosts=[])
    try:
        assert socket.socket.connect.__name__ == "guarded_connect"
        # Public target: must be blocked BEFORE any real connection.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocked = False
            try:
                s.connect(("8.8.8.8", 53))
            except airgap.EgressBlockedError:
                blocked = True
            assert blocked, "public connect should raise EgressBlockedError"
        finally:
            s.close()
        # DNS resolution of a public name is blocked too.
        dns_blocked = False
        try:
            socket.getaddrinfo("huggingface.co", 443)
        except airgap.EgressBlockedError:
            dns_blocked = True
        assert dns_blocked, "public DNS resolution should be blocked"
        # Loopback: guard allows it through to the OS (closed port -> OS refuses,
        # which is NOT an EgressBlockedError -> proves loopback was permitted).
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.settimeout(0.2)
        passed_through = False
        try:
            s2.connect(("127.0.0.1", 0))
        except airgap.EgressBlockedError:
            passed_through = False
        except OSError:
            passed_through = True  # reached the OS layer (refused/invalid) -> allowed by guard
        finally:
            s2.close()
        assert passed_through, "loopback should pass the guard to the OS layer"
        print("  guard installed: public connect+DNS blocked; loopback passes through  ✓")
    finally:
        airgap.uninstall_egress_guard()
        assert socket.socket.connect.__name__ != "guarded_connect"


def test_harden_respects_allow_external_calls():
    s = Settings()
    s.security.allow_external_calls = True
    status = airgap.harden(s)
    assert status["egress_guard"] is False  # explicit opt-out honored
    # default posture installs the guard
    s2 = Settings()
    s2.security.allow_external_calls = False
    try:
        status2 = airgap.harden(s2)
        assert status2["egress_guard"] is True
    finally:
        airgap.uninstall_egress_guard()
    print("  harden(): guard on by default, off only when allow_external_calls=true  ✓")



def test_bytes_hostnames_handled():
    """Resolvers sometimes pass host as bytes (b'localhost'); must not be blocked."""
    assert airgap.should_block_address((b"localhost", 11434)) is False
    assert airgap.should_block_address((b"127.0.0.1", 11434)) is False
    assert airgap.should_block_address((b"huggingface.co", 443)) is True
    print("  bytes hostnames decoded correctly (b'localhost' allowed)  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} air-gap tests...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
