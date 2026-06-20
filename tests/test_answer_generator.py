"""Tests for the answer generator.

Purpose
-------
Exercises the full grounded-answering path with a fake retriever and fake model, so
the confidence gate, context build, prompting, citation mapping, validation, retry,
and no-answer fallback are all covered without any model download.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.llm.base import LLMResponse
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.models import RAGAnswerRequest
from backend.app.retrieval.models import HybridSearchResponse, RetrievalCandidate


def _settings():
    s = Settings()
    s.paths.metadata_db = "/tmp/__no_such_db__.db"  # isolate: no neighbor expansion
    return s


def _cand(cid, label, text, score=0.9):
    c = RetrievalCandidate(
        chunk_id=cid, document_id="d", source_system="DOCS", source_file="a.pdf",
        record_type="pdf_page", title="Page 1", citation_label=label, text=text,
    )
    c.combined_score = score
    c.vector_score = score
    c.match_reasons = ["vector", "keyword", "fusion:rrf"]
    return c


class _FakeRetriever:
    def __init__(self, response):
        self._response = response
    def search(self, request):
        return self._response


class _FakeLLM:
    """Returns scripted outputs; supports different text per attempt."""
    def __init__(self, outputs, status="ok", error=None):
        self._outputs = outputs if isinstance(outputs, list) else [outputs]
        self._i = 0
        self._status = status
        self._error = error
    async def generate(self, system_prompt, user_prompt):
        text = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        return LLMResponse(text=text, model="llama3.1:8b", status=self._status,
                           error=self._error, latency_ms=12)


def _make(gen_settings, retrieval_response, llm):
    g = RAGAnswerGenerator(gen_settings, retriever=_FakeRetriever(retrieval_response))
    g.llm_client = llm
    return g


def test_supported_answer_with_valid_citations():
    s = _settings()
    resp = HybridSearchResponse(
        query="q", confidence="high",
        results=[_cand("c1", "[DOCS: a.pdf p1]", "Alpha bravo charlie delta echo.")],
        diagnostics={"fusion_method": "rrf"},
    )
    # Model cites the short label [S1]; generator must map it to the technical label.
    llm = _FakeLLM("The answer is bravo [S1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone")
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="what is it", show_evidence=True)))
    assert out.status == "supported", (out.status, out.validation.errors)
    assert "[DOCS: a.pdf p1]" in out.answer, "short label must be mapped to technical label"
    assert out.citations == ["[DOCS: a.pdf p1]"], out.citations
    assert out.validation.valid
    print(f"  supported answer; [S1]->technical mapping; citations={out.citations}  ✓")


def test_no_answer_on_weak_retrieval():
    s = _settings()
    resp = HybridSearchResponse(query="q", confidence="low", results=[], diagnostics={})
    llm = _FakeLLM("should never be called")
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="obscure thing")))
    assert out.status == "no_answer"
    assert out.answer == s.rag.no_answer_message
    assert out.citations == []
    assert out.retrieval_diagnostics.get("no_answer_reason") == "weak_retrieval"
    print("  weak retrieval -> clean no-answer (LLM not consulted)  ✓")


def test_no_answer_fallback_on_invalid_citations():
    s = _settings()
    resp = HybridSearchResponse(
        query="q", confidence="high",
        results=[_cand("c1", "[DOCS: a.pdf p1]", "Alpha bravo charlie.")],
        diagnostics={},
    )
    # Both attempts cite a non-existent label -> invalid -> no-answer fallback.
    llm = _FakeLLM([
        "Fabricated claim [S7].\n\n### Confidence\nHigh\n\n### Missing Information\nNone",
        "Still fabricated [S7].\n\n### Confidence\nHigh\n\n### Missing Information\nNone",
    ])
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="x")))
    assert out.status == "no_answer", out.status
    assert out.retrieval_diagnostics.get("no_answer_reason") == "invalid_citations"
    assert out.used_retry is True
    print("  invalid citations after retry -> no-answer fallback (used_retry=True)  ✓")


def test_retry_recovers_valid_answer():
    s = _settings()
    resp = HybridSearchResponse(
        query="q", confidence="high",
        results=[_cand("c1", "[DOCS: a.pdf p1]", "Alpha bravo charlie.")],
        diagnostics={},
    )
    # First attempt invalid, retry valid.
    llm = _FakeLLM([
        "No citation here at all.\n\n### Confidence\nHigh\n\n### Missing Information\nNone",
        "Now grounded [S1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone",
    ])
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="x")))
    assert out.status == "supported", (out.status, out.validation.errors)
    assert out.used_retry is True
    assert out.citations == ["[DOCS: a.pdf p1]"]
    print("  first attempt invalid, retry recovers a valid cited answer  ✓")


def test_llm_error_surfaces():
    s = _settings()
    resp = HybridSearchResponse(
        query="q", confidence="high",
        results=[_cand("c1", "[DOCS: a.pdf p1]", "Alpha bravo charlie.")],
        diagnostics={},
    )
    llm = _FakeLLM("", status="error", error="connection refused")
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="x")))
    assert out.status == "error"
    assert out.error == "connection refused"
    print("  LLM transport error -> status='error' surfaced cleanly  ✓")


def test_prior_evidence_enables_followup():
    s = _settings()
    # Fresh retrieval is weak, but a prior cited chunk exists (follow-up turn).
    resp = HybridSearchResponse(query="draft it", confidence="low", results=[], diagnostics={})
    prior = [{
        "citation_label": "[DOCS: a.pdf p1]", "chunk_id": "c1", "document_id": "d",
        "source_system": "DOCS", "source_file": "a.pdf", "record_type": "pdf_page",
        "title": "Page 1", "combined_score": 0.8, "match_reasons": ["prior_evidence"],
        "text_preview": "Alpha bravo charlie delta.",
    }]
    llm = _FakeLLM("Draft based on prior [S1].\n\n### Confidence\nMedium\n\n### Missing Information\nNone")
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="draft it", prior_evidence=prior, show_evidence=True)))
    assert out.status == "supported", (out.status, out.validation.errors)
    assert out.citations == ["[DOCS: a.pdf p1]"]
    print("  weak fresh retrieval + prior evidence -> follow-up still answers  ✓")


def test_response_dict_shape_matches_ui_contract():
    """The UI reads specific keys; verify to_dict() still produces them."""
    s = _settings()
    resp = HybridSearchResponse(
        query="q", confidence="high",
        results=[_cand("c1", "[DOCS: a.pdf p1]", "Alpha bravo charlie.")], diagnostics={},
    )
    llm = _FakeLLM("Answer [S1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone")
    g = _make(s, resp, llm)
    out = asyncio.run(g.answer(RAGAnswerRequest(query="x", show_evidence=True)))
    d = out.to_dict(include_evidence=True)
    for key in ("query", "status", "confidence", "model", "answer", "citations",
                "validation", "retrieval_diagnostics", "llm_latency_ms", "evidence"):
        assert key in d, f"missing UI contract key: {key}"
    assert isinstance(d["validation"], dict) and "valid" in d["validation"]
    print(f"  response dict carries all UI-contract keys: {sorted(d.keys())}  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} answer-generator tests (async, no model)...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
