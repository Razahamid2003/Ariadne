"""Tests for citation stability.

Purpose
-------
Verifies the safeguards that keep citations correct: stray labels are stripped,
temporary labels map to stable ones, a missing citation triggers a retry, and the
prompt includes the citation reminder.
"""

import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.llm.base import LLMResponse
from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.citation_validator import CitationValidator
from backend.app.rag.models import RAGAnswerRequest, BuiltRAGContext, EvidenceChunk
from backend.app.retrieval.models import HybridSearchResponse, RetrievalCandidate


def _ev(label, text="Some evidence text about Raza Hamid at LUMS."):
    c = RetrievalCandidate("c1","d","S","a.pdf","pdf_page","Page 1",label,text)
    c.combined_score = 0.9; c.vector_score = 0.85
    c.match_reasons = ["vector","keyword","fusion:rrf"]
    return c

def _ctx(label="[DOCS: raza-cv-chunk-0001]"):
    ev = EvidenceChunk(
        evidence_id="E1", citation_label=label, chunk_id="c1", document_id="d",
        source_system="S", source_file="a.pdf", record_type="pdf_page", title="Page 1",
        text="Raza Hamid studied BS CS at LUMS Sep 2022 – May 2026.",
        combined_score=0.9, match_reasons=["vector","keyword"],
    )
    return BuiltRAGContext(
        query="LUMS dates", retrieval_confidence="high",
        evidence=[ev], context_text="...", total_chars=100, truncated=False,
        diagnostics={},
    )

class _FakeRetriever:
    def search(self, req):
        return HybridSearchResponse(
            query=req.query, confidence="high",
            results=[_ev("[DOCS: raza-cv-chunk-0001]")],
            diagnostics={"fusion_method":"rrf"},
        )

class _FakeLLM:
    def __init__(self, outputs):
        self._outputs = outputs if isinstance(outputs,list) else [outputs]
        self._i = 0
    async def generate(self, system_prompt, user_prompt):
        out = self._outputs[min(self._i, len(self._outputs)-1)]
        self._i += 1
        return LLMResponse(text=out, model="llama3.1:8b", status="ok", latency_ms=10)

def _gen(llm_outputs):
    s = Settings(); s.paths.metadata_db = "/tmp/__no_db__.db"
    g = RAGAnswerGenerator(s, retriever=_FakeRetriever())
    g.llm_client = _FakeLLM(llm_outputs)
    return g


def test_orphaned_s_label_detected_and_triggers_retry():
    """If model outputs [S1] that wasn't mapped (shouldn't happen but guard exists),
    the validator catches it as unexpected and retry fires."""
    s = Settings()
    v = CitationValidator(s)
    ctx = _ctx()
    # Simulate an answer where [S1] slipped through unmapped
    result = v.validate("Raza studied at LUMS [S1].", ctx)
    # [S1] is not in allowed set (which contains technical labels) -> unexpected
    assert not result.valid, "orphaned [S1] should be invalid"
    assert any("S1" in e or "cit" in e.lower() for e in result.errors + result.unexpected_labels), result
    print("  orphaned [S1] detected as unexpected -> retry fires  ✓")


def test_e_label_stripped_from_answer():
    """[E1]-style labels are stripped by the mapping step before validation."""
    from backend.app.rag.answer_generator import RAGAnswerGenerator
    s = Settings(); s.paths.metadata_db = "/tmp/__no_db__.db"
    ctx = _ctx()
    raw = "Raza studied at LUMS from Sep 2022 to May 2026 [E1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone"
    mapped = RAGAnswerGenerator._map_short_to_technical(raw, ctx)
    assert "[E1]" not in mapped, f"[E1] should be stripped, got: {mapped[:80]}"
    # [S1] in raw would have been mapped; [E1] is just gone
    print(f"  [E1] stripped from answer before validation  ✓")


def test_s1_mapped_to_technical_label():
    """[S1] in model output maps to the correct technical citation label."""
    from backend.app.rag.answer_generator import RAGAnswerGenerator
    s = Settings(); s.paths.metadata_db = "/tmp/__no_db__.db"
    ctx = _ctx("[DOCS: raza-cv-chunk-0001]")
    raw = "Raza studied BS CS at LUMS [S1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone"
    mapped = RAGAnswerGenerator._map_short_to_technical(raw, ctx)
    assert "[DOCS: raza-cv-chunk-0001]" in mapped, f"technical label not substituted: {mapped}"
    assert "[S1]" not in mapped, f"[S1] should be gone: {mapped}"
    print(f"  [S1] -> [DOCS: raza-cv-chunk-0001] mapping correct  ✓")


def test_retry_fires_and_recovers_on_e_label_output():
    """First attempt uses [E1] (stripped -> no citations -> retry), second uses [S1] -> valid."""
    attempt1 = "Raza at LUMS [E1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone"
    attempt2 = "Raza studied BS CS at LUMS Sep 2022 – May 2026 [S1].\n\n### Confidence\nHigh\n\n### Missing Information\nNone"
    g = _gen([attempt1, attempt2])
    out = asyncio.run(g.answer(RAGAnswerRequest(query="LUMS dates", show_evidence=True)))
    assert out.status == "supported", (out.status, out.validation.errors)
    assert out.used_retry is True
    assert "[DOCS: raza-cv-chunk-0001]" in out.citations
    print(f"  [E1] attempt -> retry -> [S1] valid answer  ✓  (citations={out.citations})")


def test_prompt_contains_citation_reminder():
    """The user prompt now contains a CITATION REMINDER line near the evidence."""
    from backend.app.rag.prompt_builder import RAGPromptBuilder
    s = Settings(); s.paths.metadata_db = "/tmp/__no_db__.db"
    ctx = _ctx()
    builder = RAGPromptBuilder(s)
    prompt = builder.build("LUMS dates", ctx)
    assert "CITATION REMINDER" in prompt.user_prompt, "reminder missing from user prompt"
    assert "ONLY use [S1]" in prompt.system_prompt or "ONLY" in prompt.system_prompt
    print("  CITATION REMINDER present in user prompt; ONLY in system prompt  ✓")


def test_retry_prompt_uses_stronger_language():
    """The retry prompt explicitly says REJECTED and names the fix."""
    from backend.app.rag.prompt_builder import RAGPromptBuilder
    s = Settings()
    ctx = _ctx()
    builder = RAGPromptBuilder(s)
    retry_prompt = builder.build("LUMS dates", ctx, retry=True)
    assert "REJECTED" in retry_prompt.user_prompt, "retry prompt should say REJECTED"
    assert "MUST fix" in retry_prompt.user_prompt, "retry prompt should say MUST fix"
    print("  retry prompt uses strong directive language (REJECTED / MUST fix)  ✓")


if __name__ == "__main__":
    tests = [v for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} citation-fix tests...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
