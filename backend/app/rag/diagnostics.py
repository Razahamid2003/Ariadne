"""Response diagnostics helpers.

Purpose
-------
Small helpers that attach consistent diagnostic fields to answer responses so the
UI and tools can show how an answer was produced.
"""

from __future__ import annotations
from typing import Any

_MULTIHOP_KEYS = {
    "safe_multihop", "safe_multihop_hops", "safe_multihop_bridge_terms",
    "safe_multihop_initial_chunks", "safe_multihop_followup_chunks",
    "safe_multihop_merged_chunks", "safe_multihop_followup_diagnostics",
    "bridge_terms", "followup_diagnostics",
}

def strip_multihop_diagnostics(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            k = str(key)
            if k in _MULTIHOP_KEYS or k.startswith("safe_multihop"):
                continue
            out[k] = strip_multihop_diagnostics(nested)
        return out
    if isinstance(value, list):
        return [strip_multihop_diagnostics(item) for item in value]
    return value

def single_pass_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    clean = strip_multihop_diagnostics(diagnostics or {})
    if not isinstance(clean, dict):
        clean = {}
    clean["single_pass_rag"] = True
    clean["automatic_multihop_disabled"] = True
    return clean
