#!/usr/bin/env python3
"""Quality and stress evaluation runner.

Purpose
-------
Runs answer-quality and stress suites against a live local instance, with options
for repeated iterations, concurrency, and a written report.

Usage
-----
    python tools/rag_eval.py --suite tests/rag_quality_suite.yaml --write-report
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.eval.answer_checks import check_answer_response  # noqa: E402
from backend.app.eval.api_client import AriadneApiClient  # noqa: E402
from backend.app.eval.check_utils import extract_source_names_from_chat, extract_source_names_from_search  # noqa: E402
from backend.app.eval.report_writer import write_markdown_report  # noqa: E402
from backend.app.eval.retrieval_checks import check_search_response  # noqa: E402
from backend.app.eval.suite_loader import load_suite  # noqa: E402


@dataclass(frozen=True)
class RunConfig:
    api_url: str
    timeout: float
    iterations: int
    concurrency: int
    run_search: bool
    run_chat: bool
    fail_on_warning: bool


def _merged_options(defaults: dict[str, Any], test: dict[str, Any], key: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(defaults.get(key), dict):
        merged.update(defaults[key])
    if isinstance(test.get(key), dict):
        merged.update(test[key])
    return merged


def _chat_payload(defaults: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "top_k": defaults.get("top_k", 12),
        "show_evidence": defaults.get("show_evidence", True),
        "answer_mode": defaults.get("answer_mode", "balanced"),
        "preview_chars": defaults.get("preview_chars", 1500),
    }
    payload.update(test.get("request") or {})
    payload.update(test.get("chat_request") or {})
    return payload


def _search_payload(defaults: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "top_k": defaults.get("top_k", 12),
        "preview_chars": defaults.get("preview_chars", 1500),
    }
    payload.update(test.get("request") or {})
    payload.update(test.get("search_request") or {})
    # /api/search does not accept answer-only fields.
    payload.pop("show_evidence", None)
    payload.pop("answer_mode", None)
    payload.pop("chat_id", None)
    return payload


def run_one(test: dict[str, Any], suite_defaults: dict[str, Any], config: RunConfig, iteration: int) -> dict[str, Any]:
    client = AriadneApiClient(config.api_url, timeout=config.timeout)
    query = str(test["query"])
    failures: list[str] = []
    warnings: list[str] = []
    search_response: dict[str, Any] | None = None
    chat_response: dict[str, Any] | None = None
    search_latency_ms: int | None = None
    chat_latency_ms: int | None = None

    if config.run_search and test.get("run_search", True):
        result = client.search(query, **_search_payload(suite_defaults, test))
        search_latency_ms = result.latency_ms
        if not result.ok:
            failures.append(f"search API failed: {result.error}")
            search_response = result.payload
        else:
            search_response = result.payload
            expectations = _merged_options(suite_defaults, test, "search_expect")
            passed, check_failures, check_warnings = check_search_response(search_response, expectations)
            if not passed:
                failures.extend(f"search: {item}" for item in check_failures)
            warnings.extend(f"search: {item}" for item in check_warnings)

    if config.run_chat and test.get("run_chat", True):
        result = client.chat(query, **_chat_payload(suite_defaults, test))
        chat_latency_ms = result.latency_ms
        if not result.ok:
            failures.append(f"chat API failed: {result.error}")
            chat_response = result.payload
        else:
            chat_response = result.payload
            expectations = _merged_options(suite_defaults, test, "answer_expect")
            passed, check_failures, check_warnings = check_answer_response(chat_response, expectations)
            if not passed:
                failures.extend(f"answer: {item}" for item in check_failures)
            warnings.extend(f"answer: {item}" for item in check_warnings)

    if config.fail_on_warning and warnings:
        failures.extend(f"warning treated as failure: {item}" for item in warnings)

    status = chat_response.get("status") if isinstance(chat_response, dict) else None
    confidence = chat_response.get("confidence") if isinstance(chat_response, dict) else None
    citations = chat_response.get("citations") if isinstance(chat_response, dict) else []
    source_documents = extract_source_names_from_chat(chat_response or {})
    search_top_sources = extract_source_names_from_search(search_response or {})[:8]

    return {
        "id": test.get("id"),
        "iteration": iteration,
        "query": query,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "status": status,
        "confidence": confidence,
        "citations": citations,
        "source_documents": source_documents,
        "search_top_sources": search_top_sources,
        "chat_latency_ms": chat_latency_ms,
        "search_latency_ms": search_latency_ms,
        "retrieval_diagnostics": (chat_response or {}).get("retrieval_diagnostics") if isinstance(chat_response, dict) else None,
        "search_diagnostics": (search_response or {}).get("diagnostics") if isinstance(search_response, dict) else None,
    }


def _expanded_jobs(tests: list[dict[str, Any]], iterations: int) -> list[tuple[dict[str, Any], int]]:
    jobs: list[tuple[dict[str, Any], int]] = []
    for iteration in range(1, iterations + 1):
        for test in tests:
            jobs.append((test, iteration))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ariadne RAG quality/stress tests.")
    parser.add_argument("--suite", default="tests/rag_quality_suite.yaml", help="Path to JSON/YAML suite file.")
    parser.add_argument("--api", default="http://127.0.0.1:8080", help="Ariadne API base URL.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--iterations", type=int, default=1, help="Run each test N times.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent test workers.")
    parser.add_argument("--no-search", action="store_true", help="Skip /api/search retrieval preflight.")
    parser.add_argument("--no-chat", action="store_true", help="Skip /api/chat answer checks.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat warnings as failures.")
    parser.add_argument("--write-report", action="store_true", help="Write a Markdown report under reports/.")
    parser.add_argument("--report-dir", default="reports", help="Directory for Markdown reports.")
    parser.add_argument("--json-out", default=None, help="Optional path for raw JSON results.")
    parser.add_argument("--only", default=None, help="Comma-separated test ids to run.")
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.no_search and args.no_chat:
        parser.error("cannot use both --no-search and --no-chat")

    suite = load_suite(args.suite)
    tests = list(suite["tests"])
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        tests = [test for test in tests if str(test.get("id")) in wanted]
        if not tests:
            raise SystemExit(f"No tests matched --only={args.only}")

    config = RunConfig(
        api_url=args.api,
        timeout=args.timeout,
        iterations=args.iterations,
        concurrency=args.concurrency,
        run_search=not args.no_search,
        run_chat=not args.no_chat,
        fail_on_warning=args.fail_on_warning,
    )

    started = time.perf_counter()
    jobs = _expanded_jobs(tests, args.iterations)
    results: list[dict[str, Any]] = []

    if args.concurrency == 1:
        for index, (test, iteration) in enumerate(jobs, start=1):
            result = run_one(test, suite.get("defaults") or {}, config, iteration)
            results.append(result)
            print(f"[{index}/{len(jobs)}] {'PASS' if result['passed'] else 'FAIL'} {result['id']} ({result['chat_latency_ms'] or result['search_latency_ms']} ms)")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_map = {
                executor.submit(run_one, test, suite.get("defaults") or {}, config, iteration): (test, iteration)
                for test, iteration in jobs
            }
            for index, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(jobs)}] {'PASS' if result['passed'] else 'FAIL'} {result['id']} ({result['chat_latency_ms'] or result['search_latency_ms']} ms)")

    elapsed = int((time.perf_counter() - started) * 1000)
    total = len(results)
    passed = sum(1 for item in results if item.get("passed"))
    failed = total - passed

    meta = {
        "suite_name": suite.get("suite_name"),
        "api_url": args.api,
        "iterations": args.iterations,
        "concurrency": args.concurrency,
        "elapsed_ms": elapsed,
    }

    payload = {"meta": meta, "results": results}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = None
    if args.write_report:
        report_path = write_markdown_report(
            report_dir=args.report_dir,
            suite_name=str(suite.get("suite_name")),
            results=results,
            meta=meta,
        )

    print("")
    print(f"Summary: {passed}/{total} passed, {failed} failed, elapsed {elapsed} ms")
    if report_path:
        print(f"Report: {report_path}")

    if failed:
        print("")
        print("Failures:")
        for item in results:
            if not item.get("passed"):
                print(f"- {item.get('id')}: {item.get('query')}")
                for failure in item.get("failures", []):
                    print(f"  - {failure}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
