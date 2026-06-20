"""Retrieval-level evaluation checks.

Purpose
-------
Scores a single search response against a test case: whether the expected sources
appear in the top results and whether forbidden ones are absent.
"""

from __future__ import annotations

from typing import Any

from backend.app.eval.check_utils import contains_any, extract_source_names_from_search, missing_all_terms, present_terms, text_blob


def _top_results(response: dict[str, Any], top_n: int | None = None) -> list[dict[str, Any]]:
    results = response.get("results") or []
    if not isinstance(results, list):
        return []
    clean = [item for item in results if isinstance(item, dict)]
    return clean[:top_n] if top_n else clean


def check_search_response(response: dict[str, Any], expectations: dict[str, Any] | None) -> tuple[bool, list[str], list[str]]:
    """Return (passed, failures, warnings) for one search response."""

    expectations = expectations or {}
    failures: list[str] = []
    warnings: list[str] = []

    top_n = expectations.get("top_n")
    top = _top_results(response, int(top_n) if top_n else None)
    full_results = _top_results(response)
    blob = text_blob(top)
    all_blob = text_blob(full_results)
    source_blob = text_blob(extract_source_names_from_search({"results": top}))

    min_results = int(expectations.get("min_results") or 0)
    if len(full_results) < min_results:
        failures.append(f"expected at least {min_results} retrieved results, got {len(full_results)}")

    required_source_terms_any = expectations.get("required_source_terms_any") or []
    if required_source_terms_any and not contains_any(source_blob, required_source_terms_any):
        failures.append(f"none of the required top source terms appeared: {required_source_terms_any}")

    required_source_terms_all = expectations.get("required_source_terms_all") or []
    missing_sources = missing_all_terms(source_blob, required_source_terms_all)
    if missing_sources:
        failures.append(f"missing required top source terms: {missing_sources}")

    forbidden_source_terms_any = expectations.get("forbidden_source_terms_any") or []
    bad_sources = present_terms(source_blob, forbidden_source_terms_any)
    if bad_sources:
        failures.append(f"forbidden source terms appeared in top results: {bad_sources}")

    required_text_terms_all = expectations.get("required_text_terms_all") or []
    missing_terms = missing_all_terms(blob, required_text_terms_all)
    if missing_terms:
        failures.append(f"missing required top result text terms: {missing_terms}")

    required_text_terms_any = expectations.get("required_text_terms_any") or []
    if required_text_terms_any and not contains_any(blob, required_text_terms_any):
        failures.append(f"none of the required top result text terms appeared: {required_text_terms_any}")

    forbidden_text_terms_any = expectations.get("forbidden_text_terms_any") or []
    bad_text = present_terms(blob, forbidden_text_terms_any)
    if bad_text:
        failures.append(f"forbidden text terms appeared in top results: {bad_text}")

    # Wider recovery check: useful for diagnosing whether retrieval found evidence
    # anywhere, even if top-k/context selection later failed.
    required_anywhere_terms_any = expectations.get("required_anywhere_terms_any") or []
    if required_anywhere_terms_any and not contains_any(all_blob, required_anywhere_terms_any):
        failures.append(f"none of the required terms appeared anywhere in retrieved results: {required_anywhere_terms_any}")

    diagnostics = response.get("diagnostics") or {}
    if expectations.get("require_reranker_used") and not diagnostics.get("reranker_used"):
        failures.append(f"expected reranker_used=true, diagnostics={diagnostics}")
    if expectations.get("forbid_reranker_skip_reason") and diagnostics.get("reason"):
        failures.append(f"unexpected reranker skip reason: {diagnostics.get('reason')}")

    max_latency_ms = expectations.get("max_latency_ms")
    latency = response.get("latency_ms")
    if max_latency_ms is not None and latency is not None and float(latency) > float(max_latency_ms):
        warnings.append(f"search latency {latency} ms above target {max_latency_ms} ms")

    return not failures, failures, warnings
