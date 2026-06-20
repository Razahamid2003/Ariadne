"""Tests for fusion and retrieval.

Purpose
-------
Exercises the fusion math and the full search wiring with fabricated candidates and
mocked collaborators, so they run without any model: agreement is rewarded, exact
IDs surface via keyword search, scores normalize, and confidence reflects absolute
match strength.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import Settings
from backend.app.retrieval.fusion import (
    estimate_confidence,
    fuse_candidates,
    min_max_normalize,
    reciprocal_rank_fusion_score,
)
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.models import HybridSearchRequest, RetrievalCandidate


def _cand(chunk_id, *, vr=None, kr=None, vs=0.0, ks=0.0, rt="pdf_page", text="", title="", src=""):
    c = RetrievalCandidate(
        chunk_id=chunk_id, document_id="d", source_system="S", source_file=src,
        record_type=rt, title=title, citation_label=f"[{chunk_id}]", text=text,
    )
    c.vector_rank, c.keyword_rank, c.vector_score, c.keyword_score = vr, kr, vs, ks
    return c


def test_rrf_rewards_agreement():
    """A doc ranked well by BOTH retrievers should beat docs found by only one."""
    both = reciprocal_rank_fusion_score(1, 1, rrf_k=60, vector_weight=0.6, keyword_weight=0.4)
    vonly = reciprocal_rank_fusion_score(1, None, rrf_k=60, vector_weight=0.6, keyword_weight=0.4)
    konly = reciprocal_rank_fusion_score(None, 1, rrf_k=60, vector_weight=0.6, keyword_weight=0.4)
    assert both > vonly > konly  # agreement wins; vector weighted higher than keyword
    print(f"  RRF both={both:.5f}  vector-only={vonly:.5f}  keyword-only={konly:.5f}  ✓")


def test_exact_id_surfaces_via_keyword():
    """Realistic 'EMP-102' case: BM25 ranks the exact row #1 and the vector index
    also retrieves it (surrounding text is semantically close), so it is found by
    BOTH retrievers and wins on RRF -- with no hardcoded exact-match boost.

    This also documents RRF's core property: agreement between retrievers is
    rewarded. A document confirmed by both beats one found by a single retriever.
    """
    cands = [
        _cand("exact_emp102", vr=4, kr=1, rt="csv_row"),     # both; keyword loves the exact ID
        _cand("both_competitor", vr=1, kr=6, rt="pdf_page"), # both; vector loves a paraphrase
        _cand("single_noise", vr=2, kr=None, rt="pdf_page"), # one retriever only
    ]
    fuse_candidates(cands, method="rrf", rrf_k=60, vector_weight=0.6, keyword_weight=0.4,
                    record_type_weights=None, apply_record_type_prior=False)
    ranked = sorted(cands, key=lambda c: c.combined_score, reverse=True)
    ids = [c.chunk_id for c in ranked]
    assert ids[0] == "exact_emp102", ids
    assert ids[-1] == "single_noise", ids  # single-retriever hit ranks below both-retriever hits
    print(f"  exact-id wins via both-retriever agreement (no boost): {ids}  ✓")


def test_scores_normalized_0_1():
    cands = [_cand("a", vr=1, kr=1), _cand("b", vr=10, kr=None), _cand("c", vr=None, kr=20)]
    fuse_candidates(cands, method="rrf", rrf_k=60, vector_weight=0.6, keyword_weight=0.4,
                    record_type_weights=None, apply_record_type_prior=False)
    scores = [c.combined_score for c in cands]
    assert min(scores) == 0.0 and max(scores) == 1.0
    assert all(0.0 <= s <= 1.0 for s in scores)
    print(f"  fused scores normalized to 0..1: min={min(scores)} max={max(scores)}  ✓")


def test_min_max_edge_cases():
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0]) == [1.0]
    assert min_max_normalize([0.0, 0.0]) == [0.0, 0.0]
    print("  min-max edge cases (empty / single / constant)  ✓")


def test_record_type_prior_is_transparent():
    """OCR text should be gently discounted vs native, and it must be logged."""
    weights = {"pdf_page": 1.0, "pdf_page_ocr_text": 0.5}
    a = _cand("native", vr=1, kr=1, rt="pdf_page")
    b = _cand("ocr", vr=1, kr=1, rt="pdf_page_ocr_text")
    fuse_candidates([a, b], method="rrf", rrf_k=60, vector_weight=0.6, keyword_weight=0.4,
                    record_type_weights=weights, apply_record_type_prior=True)
    assert a.combined_score > b.combined_score
    assert any(r.startswith("record_type_prior:") for r in b.match_reasons), b.match_reasons
    print(f"  record-type prior applied + logged (native {a.combined_score:.2f} > ocr {b.combined_score:.2f})  ✓")


def test_confidence_uses_absolute_signal():
    """Confidence must reflect ABSOLUTE match strength, not normalized rank.
    A batch where the best is normalized to 1.0 but cosine is tiny -> low."""
    # All irrelevant: high normalized combined_score but low cosine -> must be low
    weak = [_cand("a", vs=0.12), _cand("b", vs=0.08)]
    weak[0].combined_score, weak[1].combined_score = 1.0, 0.0  # normalized says "great"
    assert estimate_confidence(weak, rerank_used=False) == "low", "irrelevant batch must be low"

    strong = [_cand("a", vs=0.72), _cand("b", vs=0.61)]
    assert estimate_confidence(strong, rerank_used=True) == "high"

    med = [_cand("a", vs=0.40)]
    assert estimate_confidence(med, rerank_used=False) == "medium"
    assert estimate_confidence([], rerank_used=False) == "low"
    print("  confidence keyed to absolute cosine/rerank signal, not normalized rank  ✓")


# ---- full search() wiring with mocked collaborators (no torch) ----
class _FakeEmbedder:
    def encode(self, texts, show_progress_bar=False):
        class B: pass
        b = B(); b.vectors = np.zeros((1, 8), dtype=np.float32); return b

class _FakeVectorIndex:
    def search(self, query_vector, top_k, source_system=None, record_type=None):
        return [
            {"chunk_id": "v1", "document_id": "d1", "source_system": "S", "source_file": "a.pdf",
             "record_type": "pdf_page", "title": "Page 1", "citation_label": "[v1]",
             "text": "alpha bravo", "score": 0.91},
            {"chunk_id": "shared", "document_id": "d2", "source_system": "S", "source_file": "b.pdf",
             "record_type": "pdf_page", "title": "Page 2", "citation_label": "[shared]",
             "text": "charlie delta", "score": 0.80},
        ]

class _FakeKeywordIndex:
    def search(self, query, top_k, source_system=None, record_type=None):
        return [
            RetrievalCandidate("shared", "d2", "S", "b.pdf", "pdf_page", "Page 2", "[shared]",
                               "charlie delta", keyword_score=7.2),
            RetrievalCandidate("k1", "d3", "S", "c.csv", "csv_row", "Row 5", "[k1]",
                               "EMP-102 echo", keyword_score=6.1),
        ]

class _FakeReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k], {"reranker_enabled": False, "reranker_used": False}


def test_full_search_wiring():
    s = Settings()
    r = HybridRetriever(s)
    r._embedding_model = _FakeEmbedder()
    r._vector_index = _FakeVectorIndex()
    r._keyword_index = _FakeKeywordIndex()
    r._reranker = _FakeReranker()

    resp = r.search(HybridSearchRequest(query="who is EMP-102", top_k=5))
    ids = [c.chunk_id for c in resp.results]
    assert "shared" in ids  # found by both retrievers
    assert resp.diagnostics["fusion_method"] == "rrf"
    assert resp.diagnostics["merged_candidates"] == 3  # v1, shared, k1
    # 'shared' (both retrievers) outranks single-retriever hits
    assert ids[0] == "shared", f"expected 'shared' first, got {ids}"
    # k1 is the normalized-zero tail -> trimmed by default min_score (0.12)
    assert "k1" not in ids, f"k1 should be tail-trimmed by min_score, got {ids}"
    print(f"  full search() wiring: results={ids}  conf={resp.confidence}  (k1 tail-trimmed)  ✓")

    # min_score is configurable: drop it to 0 and the tail returns
    s2 = Settings()
    s2.retrieval.min_score = 0.0
    r2 = HybridRetriever(s2)
    r2._embedding_model = _FakeEmbedder(); r2._vector_index = _FakeVectorIndex()
    r2._keyword_index = _FakeKeywordIndex(); r2._reranker = _FakeReranker()
    resp2 = r2.search(HybridSearchRequest(query="who is EMP-102", top_k=5))
    assert len(resp2.results) == 3, [c.chunk_id for c in resp2.results]
    print(f"  min_score configurable: min_score=0 -> {len(resp2.results)} results  ✓")


def test_empty_query():
    r = HybridRetriever(Settings())
    resp = r.search(HybridSearchRequest(query="   ", top_k=5))
    assert resp.results == [] and resp.confidence == "low"
    print("  empty query -> safe empty low-confidence response  ✓")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} retrieval-engine tests (no model runtime)...\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n✓ ALL {len(tests)} TESTS PASSED")
