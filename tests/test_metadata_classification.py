"""Tests for document-type classification and the metadata signal.

Purpose
-------
Covers the whole metadata path with a fake model: document classification, query
intent, and the soft ranking nudge that promotes matching types without filtering
anything out.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ingestion.document_classifier import (
    MetadataField, build_classification_prompt, classify_document, parse_classification,
)
from backend.app.retrieval.query_intent import classify_query_intent, parse_intent
from backend.app.retrieval.fusion import apply_document_type_nudge
from backend.app.retrieval.models import RetrievalCandidate


FIELDS = [MetadataField(
    name="document_type",
    description="Kind of document.",
    allowed_values=("resume", "form", "tabular_record", "product_brochure", "other"),
    value_descriptions={"resume": "A CV of one person."},
    fallback="other",
)]


class _FakeLLM:
    def __init__(self, text, status="ok"):
        self._text, self._status = text, status
    async def generate(self, system_prompt, user_prompt):
        class R: pass
        r = R(); r.text = self._text; r.status = self._status; r.error = None; r.latency_ms = 5
        return r


def _cand(cid, doc_type=None, score=0.5):
    c = RetrievalCandidate(cid, "d", "S", "a.pdf", "pdf_page", "T", f"[{cid}]", "text")
    c.combined_score = score
    if doc_type is not None:
        c.metadata_json = json.dumps({"document_type": doc_type})
    return c


# ---- ingestion-side classification ----
def test_parse_classification_validates_allowed_values():
    out = parse_classification('{"document_type": "resume"}', FIELDS)
    assert out["document_type"] == "resume"
    print("  valid allowed value parsed  ✓")


def test_parse_classification_coerces_close_variant():
    out = parse_classification('{"document_type": "Resume "}', FIELDS)
    assert out["document_type"] == "resume"
    print("  close variant coerced to canonical value  ✓")


def test_parse_classification_falls_back_on_unknown():
    out = parse_classification('{"document_type": "spaceship_manual"}', FIELDS)
    assert out["document_type"] == "other", out
    print("  unknown value -> fallback 'other'  ✓")


def test_parse_classification_handles_garbage():
    assert parse_classification("not json at all", FIELDS)["document_type"] == "other"
    assert parse_classification("", FIELDS)["document_type"] == "other"
    print("  unparseable output -> graceful fallback  ✓")


def test_classify_document_graceful_on_llm_error():
    llm = _FakeLLM("", status="error")
    out = asyncio.run(classify_document("some cv text", FIELDS, llm))
    assert out["document_type"] == "other"
    print("  LLM error during classification -> fallback, no raise  ✓")


def test_classify_document_happy_path():
    llm = _FakeLLM('{"document_type": "resume"}')
    out = asyncio.run(classify_document("Raza Hamid\nEducation: LUMS BS CS", FIELDS, llm))
    assert out["document_type"] == "resume"
    print("  document classified as resume via LLM  ✓")


def test_classification_prompt_contains_allowed_values():
    sys_p, user_p = build_classification_prompt("sample text", FIELDS)
    assert "resume" in user_p and "product_brochure" in user_p
    assert "JSON" in sys_p
    print("  classification prompt lists allowed values + demands JSON  ✓")


# ---- query-intent classification ----
def test_parse_intent_maps_to_types():
    assert parse_intent("resume", ["resume", "form"]) == ["resume"]
    assert parse_intent("resume, form", ["resume", "form"]) == ["resume", "form"]
    print("  intent parsed to type list  ✓")


def test_parse_intent_any_and_garbage_fail_open():
    assert parse_intent("any", ["resume"]) == ["any"]
    assert parse_intent("", ["resume"]) == ["any"]
    assert parse_intent("banana", ["resume"]) == ["any"]
    print("  ambiguous/empty/unknown intent -> ['any'] (fail-open)  ✓")


def test_classify_query_intent_graceful():
    llm = _FakeLLM("", status="error")
    out = asyncio.run(classify_query_intent("who is a recent grad", ["resume"], llm))
    assert out == ["any"]
    print("  intent classification LLM error -> ['any']  ✓")


# ---- soft fusion nudge ----
def test_nudge_promotes_matching_type_without_filtering():
    # form scored higher than resume initially (the 'Under-Graduate' bug shape)
    form = _cand("form_chunk", "form", score=0.80)
    resume = _cand("resume_chunk", "resume", score=0.60)
    other = _cand("noise", "product_brochure", score=0.50)
    cands = [form, resume, other]
    apply_document_type_nudge(cands, ["resume"], nudge_weight=0.30)
    # resume should now outrank the form (soft nudge flipped the order)
    ranked = sorted(cands, key=lambda c: c.combined_score, reverse=True)
    assert ranked[0].chunk_id == "resume_chunk", [c.chunk_id for c in ranked]
    # nothing was removed
    assert len(cands) == 3
    # the non-matching form is still present with a real score
    assert form.combined_score > 0
    assert any("doctype_match:resume" in r for r in resume.match_reasons)
    print(f"  soft nudge promoted resume over form, nothing filtered: {[c.chunk_id for c in ranked]}  ✓")


def test_nudge_noop_on_any():
    a = _cand("a", "form", 0.8); b = _cand("b", "resume", 0.6)
    before = (a.combined_score, b.combined_score)
    apply_document_type_nudge([a, b], ["any"], nudge_weight=0.3)
    assert (a.combined_score, b.combined_score) == before
    print("  intent ['any'] -> nudge is a no-op  ✓")


def test_nudge_noop_when_no_types_present():
    a = _cand("a", None, 0.8); b = _cand("b", None, 0.6)  # no metadata
    before = (a.combined_score, b.combined_score)
    apply_document_type_nudge([a, b], ["resume"], nudge_weight=0.3)
    assert (a.combined_score, b.combined_score) == before
    print("  no candidate carries a type -> nudge is a no-op  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} metadata-classification tests...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
