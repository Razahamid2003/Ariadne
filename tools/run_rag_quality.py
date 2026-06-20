"""Local answer-quality harness.

Purpose
-------
Runs a small suite of questions against a running instance and checks each answer
for expected and forbidden terms.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from backend.app.core.config import load_settings
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.models import RAGAnswerRequest


def _contains_all(text: str, values: list[str]) -> bool:
    lower = text.lower()
    return all(value.lower() in lower for value in values or [])


def _contains_none(text: str, values: list[str]) -> bool:
    lower = text.lower()
    return all(value.lower() not in lower for value in values or [])


async def run_suite(config_path: str, suite_path: str, output_path: str | None = None) -> dict[str, Any]:
    settings = load_settings(config_path)
    generator = RAGAnswerGenerator(settings)
    cases = yaml.safe_load(Path(suite_path).read_text(encoding="utf-8")) or []
    results: list[dict[str, Any]] = []

    for case in cases:
        response = await generator.answer(RAGAnswerRequest(query=case["query"], top_k=int(case.get("top_k", 8))))
        answer = response.answer or ""
        checks = {
            "status": True if not case.get("expected_status") else response.status == case.get("expected_status"),
            "must_include": _contains_all(answer, case.get("must_include") or []),
            "must_not_include": _contains_none(answer, case.get("must_not_include") or []),
            "citations": True if not case.get("expected_citations") else bool(response.citations),
            "validation": response.validation.valid,
        }
        passed = all(checks.values())
        results.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "passed": passed,
                "checks": checks,
                "status": response.status,
                "confidence": response.confidence,
                "citations": response.citations,
                "error": response.error,
            }
        )

    summary = {
        "passed": sum(1 for item in results if item["passed"]),
        "failed": sum(1 for item in results if not item["passed"]),
        "total": len(results),
        "results": results,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Ariadne RAG quality checks.")
    parser.add_argument("--config", default="config/client.yaml")
    parser.add_argument("--suite", default="tests/rag_quality/questions.yaml")
    parser.add_argument("--output", default="storage/logs/rag_quality_report.json")
    args = parser.parse_args()
    summary = asyncio.run(run_suite(args.config, args.suite, args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
