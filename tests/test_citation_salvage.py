"""Tests for citation salvage.

Purpose
-------
Confirms that uncited-but-supported sentences get the right citation attached,
unsupported sentences are never given a fabricated source, and a forgotten-citation
answer is recovered end to end.
"""

import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.llm.base import LLMResponse
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.citation_salvage import salvage_citations
from backend.app.rag.models import RAGAnswerRequest, BuiltRAGContext, EvidenceChunk
from backend.app.retrieval.models import HybridSearchResponse, RetrievalCandidate


def _ev(label, text, idx=1):
    return EvidenceChunk(
        evidence_id=f"E{idx}", citation_label=label, chunk_id=f"c{idx}", document_id="d",
        source_system="DOCS", source_file="cv.pdf", record_type="pdf_page", title="CV",
        text=text, combined_score=0.5,
    )


def test_salvage_attributes_supported_sentence():
    ev = [_ev("[DOCS: raza-cv-0]", "Raza Hamid recently graduated from LUMS with a BS in Computer Science in 2026.")]
    answer = "Raza Hamid recently graduated from LUMS with a degree in Computer Science.\n\n### Confidence\nMedium\n\n### Missing Information\nNone"
    out = salvage_citations(answer, ev, min_overlap=0.4)
    assert "[DOCS: raza-cv-0]" in out, out
    # meta sections untouched
    assert "### Confidence\nMedium" in out
    print("  supported sentence gets the matching citation appended  ✓")


def test_salvage_does_not_fabricate_for_unsupported_sentence():
    ev = [_ev("[DOCS: raza-cv-0]", "Raza Hamid graduated from LUMS in Computer Science.")]
    answer = "The weather in Karachi is hot and humid in summer."
    out = salvage_citations(answer, ev, min_overlap=0.5)
    assert "[DOCS: raza-cv-0]" not in out, "must NOT attribute an unsupported sentence"
    assert out == answer
    print("  unsupported sentence left uncited (no fabrication)  ✓")


def test_salvage_leaves_already_cited_sentences_alone():
    ev = [_ev("[DOCS: raza-cv-0]", "Raza Hamid graduated from LUMS.")]
    answer = "Raza Hamid graduated from LUMS [DOCS: raza-cv-0]."
    out = salvage_citations(answer, ev, min_overlap=0.4)
    assert out.count("[DOCS: raza-cv-0]") == 1, "must not double-cite"
    print("  already-cited sentence not double-cited  ✓")


def test_salvage_picks_best_matching_chunk():
    ev = [
        _ev("[DOCS: cv-0]", "Raza Hamid studied computer science at LUMS university.", 1),
        _ev("[DOCS: thermal-0]", "The HIKVISION thermal camera has 640x512 resolution.", 2),
    ]
    answer = "Raza Hamid studied computer science at LUMS."
    out = salvage_citations(answer, ev, min_overlap=0.4)
    assert "[DOCS: cv-0]" in out and "[DOCS: thermal-0]" not in out
    print("  salvage attributes to the best-matching chunk only  ✓")


# ---- end-to-end: the "give me the profile" failure recovers via salvage ----
class _FakeRetriever:
    def search(self, req):
        c = RetrievalCandidate("c1","d","DOCS","cv.pdf","pdf_page","CV","[DOCS: raza-cv-0]",
                               "Raza Hamid is a recent LUMS graduate with a BS in Computer Science and AI MLOps experience.")
        c.combined_score = 0.6
        return HybridSearchResponse(query=req.query, confidence="high", results=[c], diagnostics={})


class _ForgetfulLLM:
    """Writes a grounded answer but never emits [S#] — even on retry."""
    async def generate(self, system_prompt, user_prompt):
        return LLMResponse(
            text="Raza Hamid is a recent LUMS graduate with a BS in Computer Science and AI MLOps experience.\n\n### Confidence\nHigh\n\n### Missing Information\nNone",
            model="llama3.1:8b", status="ok", latency_ms=10)


def test_end_to_end_salvage_recovers_uncited_answer():
    s = Settings()
    s.paths.metadata_db = "/tmp/__no_db__.db"
    s.rag.citation_salvage_enabled = True
    s.rag.citation_salvage_min_overlap = 0.4
    g = RAGAnswerGenerator(s, retriever=_FakeRetriever())
    g.llm_client = _ForgetfulLLM()
    out = asyncio.run(g.answer(RAGAnswerRequest(query="Give me the indexed profile for Raza Hamid", show_evidence=True)))
    assert out.status == "supported", (out.status, out.validation.errors)
    assert "[DOCS: raza-cv-0]" in out.citations, out.citations
    assert out.retrieval_diagnostics.get("citation_salvage_used") is True
    print(f"  end-to-end: forgetful LLM answer salvaged to supported  ✓  (citations={out.citations})")


def test_salvage_disabled_falls_back_to_no_answer():
    s = Settings()
    s.paths.metadata_db = "/tmp/__no_db__.db"
    s.rag.citation_salvage_enabled = False
    g = RAGAnswerGenerator(s, retriever=_FakeRetriever())
    g.llm_client = _ForgetfulLLM()
    out = asyncio.run(g.answer(RAGAnswerRequest(query="x")))
    assert out.status == "no_answer"
    assert out.retrieval_diagnostics.get("no_answer_reason") == "invalid_citations"
    print("  salvage disabled -> normal no-answer fallback preserved  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} citation-salvage tests...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
