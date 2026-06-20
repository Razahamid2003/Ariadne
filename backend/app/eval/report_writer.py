"""Markdown report writer for evaluation runs.

Purpose
-------
Writes a readable Markdown report summarizing a quality or stress evaluation run.

Flow
----
Takes the collected per-case results and writes a timestamped Markdown file,
returning its path.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.eval.check_utils import summarize_latencies


def _safe_filename(value: str) -> str:
    keep = []
    for char in value.lower():
        if char.isalnum():
            keep.append(char)
        elif char in {"-", "_", " ", "."}:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name or "rag_eval"


def write_markdown_report(
    *,
    report_dir: str | Path,
    suite_name: str,
    results: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    """Write a Markdown report and return its path."""

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"rag_eval_{_safe_filename(suite_name)}_{stamp}.md"

    total = len(results)
    passed = sum(1 for item in results if item.get("passed"))
    failed = total - passed
    chat_latencies = [float(item.get("chat_latency_ms")) for item in results if item.get("chat_latency_ms") is not None]
    search_latencies = [float(item.get("search_latency_ms")) for item in results if item.get("search_latency_ms") is not None]

    lines: list[str] = []
    lines.append(f"# RAG evaluation report: {suite_name}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Passed: {passed}/{total}")
    lines.append(f"- Failed: {failed}/{total}")
    lines.append(f"- API: {meta.get('api_url')}")
    lines.append(f"- Iterations: {meta.get('iterations')}")
    lines.append(f"- Concurrency: {meta.get('concurrency')}")
    lines.append("")
    lines.append("## Latency")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "chat_ms": summarize_latencies(chat_latencies),
        "search_ms": summarize_latencies(search_latencies),
    }, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Result | Test | Query | Status | Confidence | Chat ms | Search ms | Failures |")
    lines.append("|---|---|---|---|---|---:|---:|---|")
    for item in results:
        mark = "PASS" if item.get("passed") else "FAIL"
        failures = "<br>".join(str(x).replace("|", "\\|") for x in item.get("failures", [])) or ""
        query = str(item.get("query", "")).replace("|", "\\|")
        lines.append(
            f"| {mark} | `{item.get('id')}` | {query} | {item.get('status') or ''} | "
            f"{item.get('confidence') or ''} | {item.get('chat_latency_ms') or ''} | "
            f"{item.get('search_latency_ms') or ''} | {failures} |"
        )

    lines.append("")
    lines.append("## Detailed diagnostics")
    lines.append("")
    for item in results:
        lines.append(f"### {item.get('id')} — {'PASS' if item.get('passed') else 'FAIL'}")
        lines.append("")
        lines.append(f"Query: `{item.get('query')}`")
        lines.append("")
        if item.get("failures"):
            lines.append("Failures:")
            for failure in item.get("failures", []):
                lines.append(f"- {failure}")
            lines.append("")
        if item.get("warnings"):
            lines.append("Warnings:")
            for warning in item.get("warnings", []):
                lines.append(f"- {warning}")
            lines.append("")
        lines.append("```json")
        compact = {
            "status": item.get("status"),
            "confidence": item.get("confidence"),
            "citations": item.get("citations"),
            "source_documents": item.get("source_documents"),
            "retrieval_diagnostics": item.get("retrieval_diagnostics"),
            "search_top_sources": item.get("search_top_sources"),
        }
        lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
