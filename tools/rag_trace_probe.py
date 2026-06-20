#!/usr/bin/env python3
"""Retrieval/answer trace probe.

Purpose
-------
A diagnostic tool that compares live API behavior with local retrieval and context,
showing exactly what evidence was available and helping pinpoint whether an issue is
in retrieval, context building, or generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_QUERIES = [
    "when did Raza Hamid graduate from LUMS university?",
    "What was Raza Hamid's LUMS date range?",
    "Raza Hamid LUMS BS CS dates",
    "Which indexed person looks like a recent graduate?",
    "Who in the indexed data has AI or ML experience?",
    "Does Raza Hamid know how to repair jet engines?",
    "Does Raza Hamid have experience in Cybersecurity?",
]

CRITICAL_FILES = [
    "backend/app/rag/answer_generator.py",
    "backend/app/rag/context_builder.py",
    "backend/app/rag/prompt_builder.py",
    "backend/app/rag/citation_validator.py",
    "backend/app/retrieval/hybrid_retriever.py",
    "backend/app/api/chat.py",
    "backend/app/api/search.py",
]

DATE_RANGE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"(?:19|20)\d{2}\s*(?:-|–|—|to|through)\s*"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\s*(?:-|–|—|to|through)\s*(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

PEOPLE_HINT_RE = re.compile(
    r"\b(raza|hamid|lums|lahore university|computer science|bs cs|bs computer science|"
    r"graduate|graduated|recent graduate|fresh grad|ai|ml|rag|mlops|nlp|llm|cybersecurity)\b",
    re.IGNORECASE,
)

NO_ANSWER_RE = re.compile(
    r"not enough evidence|does not contain enough information|insufficient evidence|cannot answer",
    re.IGNORECASE,
)


def sha1_file(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:12]


def short(text: Any, width: int = 500) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


def post_json(url: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            parsed["_probe_http_status"] = resp.status
            parsed["_probe_latency_ms"] = int((time.perf_counter() - started) * 1000)
            return parsed
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "_probe_error": f"HTTP {exc.code}",
            "_probe_body": body,
            "_probe_latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "_probe_error": repr(exc),
            "_probe_latency_ms": int((time.perf_counter() - started) * 1000),
        }


def print_jsonish(title: str, obj: Any) -> None:
    print(f"\n--- {title} ---")
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def print_hashes(root: Path) -> None:
    print("\n=== LOCAL SOURCE HASHES ===")
    for rel in CRITICAL_FILES:
        print(f"{sha1_file(root / rel)}  {rel}")


def load_local_stack(root: Path):
    os.chdir(root)
    sys.path.insert(0, str(root))

    from backend.app.runtime.app_state import RAGSAppState  # noqa: WPS433
    from backend.app.retrieval.models import HybridSearchRequest  # noqa: WPS433

    state = RAGSAppState("config/client.yaml")
    retriever = state.get_retriever()
    generator = state.get_rag_answer_generator()
    context_builder = generator.context_builder
    return state, retriever, context_builder, HybridSearchRequest


def candidate_summary(candidate: Any, rank: int, preview_chars: int = 420) -> dict[str, Any]:
    text = getattr(candidate, "text", "") or ""
    return {
        "rank": rank,
        "chunk_id": getattr(candidate, "chunk_id", ""),
        "citation_label": getattr(candidate, "citation_label", ""),
        "source_file": getattr(candidate, "source_file", ""),
        "record_type": getattr(candidate, "record_type", ""),
        "title": getattr(candidate, "title", ""),
        "score": round(float(getattr(candidate, "combined_score", 0.0) or 0.0), 4),
        "vector_score": round(float(getattr(candidate, "vector_score", 0.0) or 0.0), 4),
        "keyword_score": round(float(getattr(candidate, "keyword_score", 0.0) or 0.0), 4),
        "reasons": list(getattr(candidate, "match_reasons", []) or []),
        "has_date_range": bool(DATE_RANGE_RE.search(text)),
        "has_people_hint": bool(PEOPLE_HINT_RE.search(text)),
        "preview": short(text, preview_chars),
    }


def evidence_summary(evidence: Any, rank: int, preview_chars: int = 700) -> dict[str, Any]:
    text = getattr(evidence, "text", "") or ""
    return {
        "rank": rank,
        "evidence_id": getattr(evidence, "evidence_id", ""),
        "chunk_id": getattr(evidence, "chunk_id", ""),
        "citation_label": getattr(evidence, "citation_label", ""),
        "source_file": getattr(evidence, "source_file", ""),
        "record_type": getattr(evidence, "record_type", ""),
        "title": getattr(evidence, "title", ""),
        "score": round(float(getattr(evidence, "combined_score", 0.0) or 0.0), 4),
        "reasons": list(getattr(evidence, "match_reasons", []) or []),
        "date_ranges": DATE_RANGE_RE.findall(text),
        "has_people_hint": bool(PEOPLE_HINT_RE.search(text)),
        "preview": short(text, preview_chars),
    }


def api_chat_trace(api_base: str, query: str, top_k: int) -> dict[str, Any]:
    return post_json(
        f"{api_base.rstrip('/')}/api/chat",
        {
            "query": query,
            "top_k": top_k,
            "show_evidence": True,
            "answer_mode": "balanced",
            "preview_chars": 2000,
        },
        timeout=240,
    )


def api_search_trace(api_base: str, query: str, top_k: int) -> dict[str, Any]:
    return post_json(
        f"{api_base.rstrip('/')}/api/search",
        {
            "query": query,
            "top_k": top_k,
            "preview_chars": 1200,
        },
        timeout=60,
    )


def local_trace(retriever: Any, context_builder: Any, request_cls: Any, query: str, top_k: int) -> dict[str, Any]:
    retrieval = retriever.search(request_cls(query=query, top_k=top_k))
    context = context_builder.build(retrieval)

    candidates = [
        candidate_summary(candidate, index + 1)
        for index, candidate in enumerate(retrieval.results)
    ]

    evidence = [
        evidence_summary(item, index + 1)
        for index, item in enumerate(context.evidence)
    ]

    return {
        "retrieval_query": retrieval.query,
        "retrieval_confidence": retrieval.confidence,
        "retrieval_diagnostics": retrieval.diagnostics,
        "retrieval_results": candidates,
        "context": {
            "evidence_count": len(context.evidence),
            "total_chars": context.total_chars,
            "truncated": context.truncated,
            "diagnostics": context.diagnostics,
            "evidence": evidence,
        },
        "flags": {
            "retrieval_has_raza_cv": any("raza hamid cv" in (c["source_file"] or "").lower() for c in candidates),
            "context_has_raza_cv": any("raza hamid cv" in (e["source_file"] or "").lower() for e in evidence),
            "retrieval_has_date_range": any(c["has_date_range"] for c in candidates),
            "context_has_date_range": any(e["date_ranges"] for e in evidence),
            "context_has_people_hint": any(e["has_people_hint"] for e in evidence),
        },
    }


def compact_api_chat(chat: dict[str, Any]) -> dict[str, Any]:
    evidence = chat.get("evidence") or []
    return {
        "http_error": chat.get("_probe_error"),
        "http_latency_ms": chat.get("_probe_latency_ms"),
        "status": chat.get("status"),
        "confidence": chat.get("confidence"),
        "citations": chat.get("citations"),
        "source_documents": [
            doc.get("display_name") or doc.get("source_file") or doc
            for doc in (chat.get("source_documents") or [])
        ],
        "used_retry": chat.get("used_retry"),
        "llm_latency_ms": chat.get("llm_latency_ms"),
        "error": chat.get("error"),
        "validation": chat.get("validation"),
        "retrieval_diagnostics": chat.get("retrieval_diagnostics"),
        "answer_preview": short(chat.get("answer"), 1200),
        "evidence_returned_count": len(evidence),
        "returned_evidence": [
            {
                "citation_label": item.get("citation_label"),
                "chunk_id": item.get("chunk_id"),
                "source_file": item.get("source_file"),
                "score": item.get("combined_score"),
                "preview": short(item.get("text_preview"), 500),
            }
            for item in evidence[:8]
        ],
    }


def compact_api_search(search: dict[str, Any]) -> dict[str, Any]:
    results = search.get("results") or []
    return {
        "http_error": search.get("_probe_error"),
        "http_latency_ms": search.get("_probe_latency_ms"),
        "confidence": search.get("confidence"),
        "result_count": search.get("result_count"),
        "diagnostics": search.get("diagnostics"),
        "top_results": [
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "source_file": item.get("source_file"),
                "score": item.get("combined_score"),
                "reasons": item.get("match_reasons"),
                "has_date_range": bool(DATE_RANGE_RE.search(item.get("text_preview") or "")),
                "preview": short(item.get("text_preview"), 350),
            }
            for item in results[:10]
        ],
    }


def diagnose(query: str, chat: dict[str, Any] | None, search: dict[str, Any] | None, local: dict[str, Any] | None) -> list[str]:
    notes: list[str] = []

    if chat:
        answer = chat.get("answer") or ""
        status = chat.get("status")
        if status == "no_answer" and local and local["flags"].get("context_has_date_range"):
            notes.append(
                "NO_ANSWER_BUT_LOCAL_CONTEXT_HAS_DATE_RANGE: retrieval/context has date evidence; failure is likely validation/no-answer/salvage, not retrieval."
            )
        if NO_ANSWER_RE.search(answer) and status == "ok":
            notes.append(
                "MIXED_OK_WITH_NO_ANSWER_WORDING: answer contains factual/ok status plus no-evidence wording."
            )
        if status == "no_answer" and re.search(r"repair jet engines|deep sea welding|changing diapers", answer, re.I):
            notes.append(
                "NO_ANSWER_ECHOES_UNSUPPORTED_QUERY: no-answer text repeats forbidden unsupported capability phrase."
            )

    if search and local:
        api_diag = search.get("diagnostics") or {}
        local_diag = local.get("retrieval_diagnostics") or {}
        api_norm = api_diag.get("query_normalized_for_retrieval")
        local_norm = local_diag.get("query_normalized_for_retrieval")
        if api_norm is not None and local_norm is not None and api_norm != local_norm:
            notes.append(
                f"API_LOCAL_NORMALIZATION_MISMATCH: API query_normalized_for_retrieval={api_norm}, local={local_norm}. Running server may be stale or different code path."
            )

    if local:
        flags = local.get("flags") or {}
        if flags.get("retrieval_has_raza_cv") and not flags.get("context_has_raza_cv"):
            notes.append(
                "RETRIEVAL_HAS_RAZA_BUT_CONTEXT_DROPPED_IT: context builder/ranking/filtering is the failure."
            )
        if flags.get("retrieval_has_date_range") and not flags.get("context_has_date_range"):
            notes.append(
                "RETRIEVAL_HAS_DATE_RANGE_BUT_CONTEXT_DROPPED_IT: context builder or truncation removed the useful span."
            )

    if not notes:
        notes.append("NO_OBVIOUS_RULE_HIT: inspect printed local context and validation fields.")
    return notes


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    queries = [args.query] if args.query else DEFAULT_QUERIES

    print(f"Project root: {root}")
    print(f"API base: {args.api_base}")
    print_hashes(root)

    state = retriever = context_builder = request_cls = None
    if not args.api_only:
        try:
            state, retriever, context_builder, request_cls = load_local_stack(root)
            print("\nLocal stack loaded: OK")
        except Exception as exc:
            print("\nLocal stack loaded: FAILED")
            print(repr(exc))
            print("Continuing with API-only trace.\n")

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        search = None
        chat = None
        local = None

        if not args.skip_search:
            search = api_search_trace(args.api_base, query, args.top_k)
            print_jsonish("API SEARCH SUMMARY", compact_api_search(search))

        if not args.skip_chat:
            chat = api_chat_trace(args.api_base, query, args.top_k)
            print_jsonish("API CHAT SUMMARY", compact_api_chat(chat))

        if retriever and context_builder and request_cls:
            try:
                local = local_trace(retriever, context_builder, request_cls, query, args.top_k)
                print_jsonish("LOCAL RETRIEVAL + CONTEXT TRACE", local)
            except Exception as exc:
                print_jsonish("LOCAL TRACE ERROR", {"error": repr(exc)})

        print_jsonish("DIAGNOSIS FLAGS", diagnose(query, chat, search, local))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--api-only", action="store_true", help="Do not import local backend modules.")
    parser.add_argument("--skip-chat", action="store_true", help="Skip slow /api/chat calls.")
    parser.add_argument("--skip-search", action="store_true", help="Skip /api/search calls.")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())