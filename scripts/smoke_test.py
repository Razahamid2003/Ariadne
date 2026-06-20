"""Deployment smoke test.

Purpose
-------
Verifies a fresh deployment end to end: configuration loads, the model is
reachable, the indexes are present, and a sample query returns a grounded answer.

Usage
-----
    python scripts/smoke_test.py --config config/client.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core import airgap

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
_failures = 0


def _mark(ok: bool | None, label: str, detail: str = "") -> None:
    global _failures
    if ok is None:
        tag = f"{YELLOW}SKIP{RESET}"
    elif ok:
        tag = f"{GREEN}PASS{RESET}"
    else:
        tag = f"{RED}FAIL{RESET}"
        _failures += 1
    print(f"  [{tag}] {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("RAGS_CONFIG_PATH", "config/client.yaml"))
    parser.add_argument("--query", default="Give a one sentence summary of the indexed data.")
    args = parser.parse_args()

    print("RAG backend smoke test\n" + "=" * 40)

    # 1. config
    try:
        from backend.app.core.config import load_settings
        settings = load_settings(args.config)
        _mark(True, "config loads", f"{args.config} | model={settings.llm.model}")
    except Exception as exc:
        _mark(False, "config loads", str(exc))
        return 1

    # 2. air-gap
    status = airgap.harden(settings)
    if settings.security.allow_external_calls:
        _mark(None, "air-gap egress guard", "allow_external_calls=true (guard intentionally off)")
    else:
        _mark(status.get("egress_guard", False), "air-gap egress guard active",
              f"approved_lan_hosts={status.get('approved_lan_hosts')}")
        _mark(airgap.should_block_address(("8.8.8.8", 53)), "public egress is blocked")
        _mark(not airgap.should_block_address(("127.0.0.1", 11434)), "loopback (Ollama) permitted")

    # 3. embeddings (must load offline)
    try:
        from backend.app.embeddings.local_embedding_model import SentenceTransformersEmbeddingModel
        emb = SentenceTransformersEmbeddingModel(settings.embeddings)
        batch = emb.encode(["smoke test sentence"], show_progress_bar=False)
        _mark(batch.dimension > 0, "embedding model loads + encodes (offline)",
              f"dim={batch.dimension}, device={settings.embeddings.device}")
    except Exception as exc:
        _mark(False, "embedding model loads + encodes", str(exc)[:160])

    # 4. reranker (optional)
    if settings.retrieval.rerank_enabled:
        try:
            from backend.app.retrieval.reranker import LocalReranker
            rr = LocalReranker(settings)
            model = rr._load_model()
            _mark(model is not None, "cross-encoder reranker loads",
                  settings.retrieval.rerank_model_name_or_path)
        except Exception as exc:
            _mark(False, "cross-encoder reranker loads", str(exc)[:160])
    else:
        _mark(None, "cross-encoder reranker", "rerank_enabled=false")

    # 5. local LLM endpoint
    try:
        from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient
        client = OpenAICompatibleLLMClient(settings.llm)
        resp = asyncio.run(client.generate(system_prompt="Reply with: ok", user_prompt="Reply with: ok"))
        _mark(resp.status == "ok", "local LLM responds",
              f"{settings.llm.base_url} model={settings.llm.model} latency={resp.latency_ms}ms"
              if resp.status == "ok" else (resp.error or "no response"))
    except Exception as exc:
        _mark(False, "local LLM responds", str(exc)[:160])

    # 6. end-to-end (only if an index exists)
    index_ready = (
        Path(settings.paths.metadata_db).exists()
        and Path(settings.vector_index.index_dir, settings.vector_index.embeddings_file).exists()
    )
    if not index_ready:
        _mark(None, "end-to-end query", "no index found — run ingestion first")
    else:
        try:
            from backend.app.rag.answer_generator import RAGAnswerGenerator
            from backend.app.rag.models import RAGAnswerRequest
            gen = RAGAnswerGenerator(settings)
            out = asyncio.run(gen.answer(RAGAnswerRequest(query=args.query, show_evidence=True)))
            _mark(out.status in {"supported", "no_answer"}, "end-to-end retrieve -> answer",
                  f"status={out.status} confidence={out.confidence} citations={len(out.citations)}")
            print(f"\n    Q: {args.query}\n    A: {out.answer[:300].strip()}")
        except Exception as exc:
            _mark(False, "end-to-end retrieve -> answer", str(exc)[:160])

    print("=" * 40)
    if _failures:
        print(f"{RED}{_failures} required check(s) failed.{RESET}")
        return 1
    print(f"{GREEN}All required checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
