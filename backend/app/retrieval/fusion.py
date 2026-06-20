"""Rank-fusion and confidence scoring.

Purpose
-------
The pure, model-free scoring math behind hybrid retrieval, kept separate from
input/output so it can be tested on its own.

What it does
------------
Combines keyword and vector rankings with reciprocal-rank fusion and weighted
scoring, applies a transparent trust prior per record type, applies a soft
document-type nudge, normalizes scores, and estimates retrieval confidence from
absolute match strength.

Flow
----
``fuse_candidates()`` assigns each candidate a combined score from its component
ranks; the document-type nudge gently lifts matches without filtering anything out;
``estimate_confidence()`` reports how strong the best matches really are so the
answer layer can decide whether to answer.
"""

from __future__ import annotations

from backend.app.retrieval.models import RetrievalCandidate


def reciprocal_rank_fusion_score(
    vector_rank: int | None,
    keyword_rank: int | None,
    *,
    rrf_k: int,
    vector_weight: float,
    keyword_weight: float,
) -> float:
    """Compute a weighted Reciprocal Rank Fusion score for one candidate.

    Each retriever in which the candidate appears contributes
    ``weight * 1 / (rrf_k + rank)``. A candidate found by both retrievers sums
    both contributions, which is what rewards agreement between dense and
    keyword search. ``rank`` is 1-based.
    """

    score = 0.0
    if vector_rank is not None:
        score += vector_weight * (1.0 / (rrf_k + vector_rank))
    if keyword_rank is not None:
        score += keyword_weight * (1.0 / (rrf_k + keyword_rank))
    return score


def min_max_normalize(values: list[float]) -> list[float]:
    """Min-max normalize a list of scores into 0..1.

    Constant or empty inputs collapse to 1.0 for any positive value, 0.0
    otherwise. This keeps a single non-empty result at full confidence rather
    than forcing it to zero.
    """

    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 if value > 0 else 0.0 for value in values]
    span = hi - lo
    return [(value - lo) / span for value in values]


def weighted_score_fusion(
    normalized_vector: float,
    normalized_keyword: float,
    *,
    vector_weight: float,
    keyword_weight: float,
) -> float:
    """Linear score fusion over already-normalized component scores."""

    return vector_weight * normalized_vector + keyword_weight * normalized_keyword


def record_type_prior(record_type: str, weights: dict[str, float], default: float = 1.0) -> float:
    """Return the transparent, query-independent trust prior for a record type.

    This is the single permitted form of "boosting": a configurable multiplier
    reflecting how much the *format* of evidence is trusted (native text vs OCR
    vs an image caption). It does not depend on the query in any way.
    """

    if not weights:
        return default
    return float(weights.get(record_type, default))


def fuse_candidates(
    candidates: list[RetrievalCandidate],
    *,
    method: str,
    rrf_k: int,
    vector_weight: float,
    keyword_weight: float,
    record_type_weights: dict[str, float] | None,
    apply_record_type_prior: bool,
) -> None:
    """Assign ``combined_score`` to every candidate in place.

    Steps:
        1. Compute a raw fused score per candidate (RRF or weighted).
        2. Multiply by the optional, transparent record-type prior.
        3. Min-max normalize the whole set into 0..1 so thresholds are stable.

    Candidates must already carry ``vector_rank``/``keyword_rank`` (for RRF) and
    ``vector_score``/``keyword_score`` (for weighted fusion).
    """

    if not candidates:
        return

    method = (method or "rrf").lower().strip()
    weights = record_type_weights or {}

    if method == "weighted":
        raw_vector = [c.vector_score for c in candidates]
        raw_keyword = [c.keyword_score for c in candidates]
        norm_vector = min_max_normalize(raw_vector)
        norm_keyword = min_max_normalize(raw_keyword)
        raw_scores = [
            weighted_score_fusion(
                nv, nk, vector_weight=vector_weight, keyword_weight=keyword_weight
            )
            for nv, nk in zip(norm_vector, norm_keyword)
        ]
    else:  # default: rrf
        raw_scores = [
            reciprocal_rank_fusion_score(
                c.vector_rank,
                c.keyword_rank,
                rrf_k=rrf_k,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            for c in candidates
        ]

    if apply_record_type_prior and weights:
        for candidate, raw in zip(candidates, raw_scores):
            prior = record_type_prior(candidate.record_type, weights)
            candidate.record_type_weight = prior
        raw_scores = [
            raw * record_type_prior(c.record_type, weights)
            for c, raw in zip(candidates, raw_scores)
        ]

    normalized = min_max_normalize(raw_scores)
    for candidate, score in zip(candidates, normalized):
        candidate.combined_score = float(score)
        candidate.add_reason(f"fusion:{method}")
        if apply_record_type_prior and weights and candidate.record_type_weight != 1.0:
            candidate.add_reason(f"record_type_prior:{candidate.record_type_weight:.2f}")


def _candidate_document_type(candidate: RetrievalCandidate) -> str:
    """Read the document_type a candidate carries (from ingestion metadata)."""

    raw = getattr(candidate, "metadata_json", None)
    if not raw:
        return ""
    try:
        import json

        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return ""
    return str(data.get("document_type", "") or "")


def apply_document_type_nudge(
    candidates: list[RetrievalCandidate],
    target_types: list[str],
    *,
    nudge_weight: float,
) -> None:
    """Softly re-rank candidates whose document_type matches the query intent.

    This is the metadata leg of retrieval, implemented
    as a SOFT, transparent signal on already-normalized 0..1 scores:

        * matching candidates get a small additive bonus, then the set is
          renormalized into 0..1;
        * NON-matching candidates are never removed or zeroed — a wrong intent
          guess can only fail to help, never hide a document.

    No-ops when intent is ['any'], weight is 0, or nothing carries a type.
    """

    if not candidates or nudge_weight <= 0:
        return
    targets = {t.lower() for t in (target_types or []) if t and t.lower() != "any"}
    if not targets:
        return

    bumped = False
    boosted: list[float] = []
    for c in candidates:
        dtype = _candidate_document_type(c).lower()
        if dtype and dtype in targets:
            boosted.append(c.combined_score + nudge_weight)
            c.add_reason(f"doctype_match:{dtype}:+{nudge_weight:.2f}")
            bumped = True
        else:
            boosted.append(c.combined_score)

    if not bumped:
        return
    for c, score in zip(candidates, min_max_normalize(boosted)):
        c.combined_score = float(score)


def _sigmoid(value: float) -> float:
    import math

    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def absolute_relevance_signal(candidate: RetrievalCandidate, *, rerank_used: bool) -> float:
    """Return an absolute (not rank-relative) relevance score in ~0..1.

    This is the signal that must drive the no-answer decision. Unlike the
    normalized ``combined_score`` (which always pins the batch-best to 1.0 even
    when every candidate is irrelevant), this reflects genuine match strength:

        * raw vector cosine similarity (embeddings are normalized, so 0..1), and
        * when a cross-encoder ran, the sigmoid of its raw relevance logit.

    The stronger of the two is used, so semantic matches and reranker-confirmed
    keyword-only matches both count.
    """

    cosine = max(0.0, float(candidate.vector_score or 0.0))
    if rerank_used and candidate.reranker_score:
        return max(cosine, _sigmoid(float(candidate.reranker_score)))
    return cosine


def estimate_confidence(
    results: list[RetrievalCandidate],
    *,
    rerank_used: bool,
    high_score: float = 0.55,
    medium_score: float = 0.35,
) -> str:
    """Estimate retrieval confidence from ABSOLUTE relevance signals.

    No keyword inspection, no query categories. Purely a function of how
    strongly the top evidence actually matches the query, so that a batch of
    uniformly-irrelevant chunks correctly yields "low" and triggers the
    no-answer fallback downstream.
    """

    if not results:
        return "low"

    signals = [absolute_relevance_signal(c, rerank_used=rerank_used) for c in results]
    top = max(signals)
    strong = len([s for s in signals[:5] if s >= medium_score])

    if top >= high_score and (rerank_used or strong >= 2):
        return "high"
    if top >= medium_score:
        return "medium"
    return "low"
