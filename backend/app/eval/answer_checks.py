"""Answer-level evaluation checks.

Purpose
-------
Scores a single chat answer against a test case's expectations.

What it does
------------
Checks that the answer status, confidence, required terms, and citations match the
expectation, and flags failure patterns such as a confident answer that also says
it has no evidence. Returns pass/fail plus failures and warnings.
"""

from __future__ import annotations

from typing import Any

from backend.app.eval.check_utils import (
    contains_any,
    extract_source_names_from_chat,
    looks_like_mixed_no_answer,
    looks_like_raw_excerpt_artifact,
    min_confidence_ok,
    missing_all_terms,
    norm,
    present_terms,
    text_blob,
)


def check_answer_response(response: dict[str, Any], expectations: dict[str, Any] | None) -> tuple[bool, list[str], list[str]]:
    """Return (passed, failures, warnings) for one chat response."""

    expectations = expectations or {}
    failures: list[str] = []
    warnings: list[str] = []

    answer = str(response.get("answer") or "")
    answer_blob = text_blob({
        "answer": response.get("answer"),
        "citations": response.get("citations"),
        "source_documents": response.get("source_documents"),
        "validation": response.get("validation"),
    })
    status = str(response.get("status") or "")
    confidence = str(response.get("confidence") or "")
    citations = response.get("citations") or []
    source_names = extract_source_names_from_chat(response)
    source_blob = text_blob(source_names)

    expected_status = expectations.get("expected_status")
    if expected_status:
        allowed = {norm(item) for item in (expected_status if isinstance(expected_status, list) else [expected_status])}
        if norm(status) not in allowed:
            failures.append(f"status {status!r} not in expected {sorted(allowed)}")

    expected_answer_type = expectations.get("expected_answer_type")
    if expected_answer_type:
        # Alias-level check so suites do not depend on one exact backend status name.
        answer_type = "no_answer" if norm(status) == "no_answer" or "not enough evidence" in norm(answer) else "answer"
        if norm(expected_answer_type) != answer_type:
            failures.append(f"answer_type {answer_type!r} != expected {expected_answer_type!r}")

    min_confidence = expectations.get("min_confidence")
    if min_confidence and not min_confidence_ok(confidence, str(min_confidence)):
        failures.append(f"confidence {confidence!r} below minimum {min_confidence!r}")

    max_confidence = expectations.get("max_confidence")
    if max_confidence:
        from backend.app.eval.check_utils import CONFIDENCE_RANK

        if CONFIDENCE_RANK.get(norm(confidence), 99) > CONFIDENCE_RANK.get(norm(max_confidence), -1):
            failures.append(f"confidence {confidence!r} above maximum {max_confidence!r}")

    if expectations.get("require_citations") and not citations:
        failures.append("expected at least one citation, but response has none")

    min_citations = int(expectations.get("min_citations") or 0)
    if min_citations and len(citations) < min_citations:
        failures.append(f"expected at least {min_citations} citations, got {len(citations)}")

    required_terms_all = expectations.get("required_terms_all") or []
    missing = missing_all_terms(answer_blob, required_terms_all)
    if missing:
        failures.append(f"missing required answer terms: {missing}")

    required_terms_any = expectations.get("required_terms_any") or []
    if required_terms_any and not contains_any(answer_blob, required_terms_any):
        failures.append(f"none of the required answer terms appeared: {required_terms_any}")

    forbidden_terms_any = expectations.get("forbidden_terms_any") or []
    present_forbidden = present_terms(answer_blob, forbidden_terms_any)
    if present_forbidden:
        failures.append(f"forbidden answer terms appeared: {present_forbidden}")

    required_source_terms_any = expectations.get("required_source_terms_any") or []
    if required_source_terms_any and not contains_any(source_blob, required_source_terms_any):
        failures.append(f"none of the required cited/source names appeared: {required_source_terms_any}")

    forbidden_source_terms_any = expectations.get("forbidden_source_terms_any") or []
    forbidden_sources = present_terms(source_blob, forbidden_source_terms_any)
    if forbidden_sources:
        failures.append(f"forbidden cited/source names appeared: {forbidden_sources}")

    if expectations.get("forbid_mixed_no_answer", True) and looks_like_mixed_no_answer(answer, status):
        failures.append("mixed no-answer detected: factual content appears together with no-evidence wording")

    if expectations.get("forbid_raw_excerpt_artifacts", True) and looks_like_raw_excerpt_artifact(answer):
        failures.append("raw excerpt artifact detected in final answer")

    validation = response.get("validation") or {}
    if isinstance(validation, dict):
        if validation.get("errors") and not expectations.get("allow_validation_errors", False):
            failures.append(f"validation errors present: {validation.get('errors')}")
        if validation.get("unexpected_labels") and not expectations.get("allow_unexpected_labels", False):
            failures.append(f"unexpected citation labels present: {validation.get('unexpected_labels')}")

    max_latency_ms = expectations.get("max_latency_ms")
    if max_latency_ms is not None:
        latency = response.get("api_latency_ms") or response.get("latency_ms")
        if latency is not None and float(latency) > float(max_latency_ms):
            warnings.append(f"latency {latency} ms above target {max_latency_ms} ms")

    return not failures, failures, warnings
