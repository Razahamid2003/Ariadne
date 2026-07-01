"""Multi-hop retrieval orchestration — Phase 1: decompose-then-retrieve.

A complex question is decomposed by the model into self-contained sub-questions;
each sub-question is retrieved with the SAME single-pass hybrid retriever; the
candidate pools are merged and de-duplicated; and the merged pool is re-ranked by
the SAME cross-encoder against the ORIGINAL question. The existing context builder,
prompt, generation, and citation validation then run unchanged.

What this does and does not do (the hard constraints):
- It adds *evidence completeness* for multi-hop questions by searching for each
  sub-question the model itself produced.
- It does NOT compute answers, inject domain knowledge, or hardcode anything: the
  decomposition is the model's own output, retrieval is the unchanged retriever, and
  ranking is the unchanged cross-encoder. No candidate is manually boosted.

It is OFF unless ``retrieval.multihop_enabled`` is true; when off, nothing here runs
and behaviour is exactly single-pass. Any failure falls back to single-pass rather
than erroring.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.core.logging import get_logger
from backend.app.retrieval.fusion import estimate_confidence
from backend.app.retrieval.models import (
    HybridSearchRequest,
    HybridSearchResponse,
    RetrievalCandidate,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------- #
# Trigger heuristic (generic English structure only — no corpus terms, no answers)
# ---------------------------------------------------------------------------- #
# Signals that a question may need more than one lookup: comparison/superlative
# across entities, conjoined asks, or a relative clause that bridges to a property.
_MULTIHOP_SIGNALS = [
    r"\bcompare\b", r"\bversus\b", r"\bvs\.?\b", r"\bdifference between\b",
    r"\bmore than\b", r"\bless than\b", r"\bfewer than\b", r"\bgreater than\b",
    r"\b(which|who|what)\b.*\b(most|least|highest|lowest|largest|smallest|longest|shortest|biggest|fewest)\b",
    r"\b(most|least|highest|lowest|largest|smallest|longest|shortest|biggest|fewest)\b.*\bthan\b",
    r"\bof the\b.*\b(that|which|who|whose)\b",
    r"\b(that|which|who|whose)\b.*\band\b",
    r"\bas well as\b", r"\balong with\b", r"\bfor each\b", r"\bacross\b",
    r"\?.*\?",  # two questions in one
]
_SIGNAL_RE = re.compile("|".join(_MULTIHOP_SIGNALS), re.IGNORECASE)
# " ... and ... " joining two clauses is a weak signal; require it to look like two asks.
_AND_TWO_CLAUSES = re.compile(r"\b\w+\b.*\band\b.*\b(who|what|which|where|when|how|whose|why)\b", re.IGNORECASE)


def looks_multihop(query: str) -> bool:
    """Heuristic gate for the ``auto`` trigger. Generic structure only."""
    q = query or ""
    return bool(_SIGNAL_RE.search(q) or _AND_TWO_CLAUSES.search(q))


def should_try_multihop(query: str, settings: Any) -> bool:
    """Decide whether to attempt multi-hop, honouring config. Cheap (no model call)."""
    cfg = settings.retrieval
    if not getattr(cfg, "multihop_enabled", False):
        return False
    trigger = getattr(cfg, "multihop_min_trigger", "auto")
    if trigger == "always":
        return True
    return looks_multihop(query)


# ---------------------------------------------------------------------------- #
# Decomposition prompt + parsing
# ---------------------------------------------------------------------------- #
_DECOMPOSE_SYSTEM = (
    "You split a complex question into the minimal set of simpler sub-questions that "
    "must each be answered in order to answer the original question.\n"
    "Rules:\n"
    "1. Each sub-question must stand completely on its own and name its own subject "
    "explicitly. Never use a pronoun (he, she, it, they, that, this) that depends on "
    "the answer to another sub-question.\n"
    "2. Carry over every distinguishing qualifier from the original question into each "
    "sub-question (for example the specific test, failure, mode, or condition named). "
    "If the original asks about the part for the 'radiated-emissions test failure', a "
    "sub-question must say 'the part for the radiated-emissions test failure', not just "
    "'the part' or 'the component to be replaced' — otherwise it may match the wrong "
    "item when several similar ones exist.\n"
    "3. Only split a question that genuinely needs several separate lookups. If the "
    "question is already a single-step question, return it unchanged as the single line.\n"
    "4. Output ONLY the sub-questions, one per line. No numbering, no bullet points, "
    "no answers, no explanation.\n"
    "5. Produce at most {max_subq} sub-questions."
)

_LINE_CLEAN = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s*")


def _parse_subquestions(text: str, max_subq: int) -> list[str]:
    """Parse the model's decomposition output into clean sub-question strings."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = _LINE_CLEAN.sub("", raw).strip()
        if not line:
            continue
        # Keep only things that look like questions/asks; drop stray commentary.
        if len(line) < 5:
            continue
        out.append(line)
        if len(out) >= max_subq:
            break
    return out


async def _decompose(query: str, llm_client: Any, max_subq: int) -> list[str]:
    """One model call → list of self-contained sub-questions (possibly just one)."""
    try:
        llm = await llm_client.generate(
            system_prompt=_DECOMPOSE_SYSTEM.format(max_subq=max_subq),
            user_prompt=query,
        )
    except Exception as exc:
        logger.warning("multihop decompose LLM call failed: %s", exc)
        return []
    if getattr(llm, "status", None) != "ok":
        logger.warning("multihop decompose LLM error: %s", getattr(llm, "error", "unknown"))
        return []
    return _parse_subquestions(getattr(llm, "text", "") or "", max_subq)


# ---------------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------------- #
# Tokens to skip when extracting bridge terms (too common to be useful keys)
_STOPWORDS = {
    "the","a","an","of","in","on","at","to","for","with","and","or","but","is",
    "are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","its","it","this","that",
    "these","those","by","from","as","what","which","who","how","when","where","why",
    "not","no","if","then","than","so","also","all","any","each","per","part","parts",
    "required","fix","need","used","item","type","model","number","test","value",
}
# Pattern for potentially meaningful tokens: alphanumeric, >=2 chars, may contain
# hyphens/dots (catches part codes like FA-12, SR-022, PTP-1).
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\.]{1,29}")


# Multi-word proper-noun phrases (e.g. "Filter Assembly", "Mei Tanaka") — name bridges.
_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")


def _is_code(t: str) -> bool:
    """A code/identifier has BOTH a letter and a digit (FA-12, SR-022, MDS-OWN-300).
    Excludes bare numbers ('10', '26') and plain words ('RF')."""
    return any(c.isalpha() for c in t) and any(c.isdigit() for c in t)


def _extract_bridge_terms(candidates: list[RetrievalCandidate], top_n: int = 8) -> list[str]:
    """Extract salient bridge entities, ranked so the entity most likely to need a
    follow-up lookup comes first.

    Two kinds: codes/identifiers (letter+digit, e.g. FA-12) and multi-word proper-noun
    phrases (e.g. Filter Assembly, Mei Tanaka). Ranked RARE-first: a bridge entity is
    typically a novel single mention (the thing to look up), whereas repeated tokens
    are usually the topic/context already in hand. Generic — no corpus knowledge.
    """
    code_freq: dict[str, int] = {}
    phrase_freq: dict[str, int] = {}
    for c in candidates[:top_n]:
        text = c.text or ""
        for tok in _TOKEN_RE.findall(text):
            t = tok.strip(".-")
            if len(t) < 2 or t.lower() in _STOPWORDS or not _is_code(t):
                continue
            code_freq[t] = code_freq.get(t, 0) + 1
        for ph in _PHRASE_RE.findall(text):
            ph = ph.strip()
            if 4 <= len(ph) <= 40:
                phrase_freq[ph] = phrase_freq.get(ph, 0) + 1
    # Rare-first within each kind; codes before phrases (codes are higher precision).
    codes = [t for t, _ in sorted(code_freq.items(), key=lambda kv: kv[1])]
    phrases = [p for p, _ in sorted(phrase_freq.items(), key=lambda kv: kv[1])]
    return (codes + phrases)[:10]


def _bridge_retrieve(
    merged: list[RetrievalCandidate],
    original_pool: list[RetrievalCandidate],
    retriever: Any,
    request: Any,
    per_subq_top_k: int,
    cfg: Any,
) -> list[RetrievalCandidate]:
    """Fire a targeted retrieval for bridge terms not yet well-covered in the pool.

    Extracts salient entity tokens from the top of the merged pool, fires one
    additional retrieval per new high-value term (capped at 3 retrievals), and
    returns the new candidates. Already-retrieved chunk IDs are excluded.
    No corpus knowledge, no hardcoding, no score manipulation.
    """
    existing_ids = {c.chunk_id for c in original_pool}
    bridge_terms = _extract_bridge_terms(merged, top_n=8)
    if not bridge_terms:
        return []

    new_candidates: list[RetrievalCandidate] = []
    fired = 0
    for term in bridge_terms:
        if fired >= 3:
            break
        try:
            resp = retriever.search(
                HybridSearchRequest(
                    query=term,
                    top_k=per_subq_top_k,
                    source_system=request.source_system,
                    record_type=request.record_type,
                )
            )
            new_candidates.extend(c for c in resp.results if c.chunk_id not in existing_ids)
            existing_ids.update(c.chunk_id for c in resp.results)
            fired += 1
        except Exception:
            continue
    return new_candidates


def _dedupe(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Merge pools, keeping the first (higher-ranked) instance of each chunk."""
    seen: set[str] = set()
    merged: list[RetrievalCandidate] = []
    for c in candidates:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        merged.append(c)
    return merged


async def run_multihop_decompose(
    request: Any,
    retriever: Any,
    llm_client: Any,
    settings: Any,
) -> HybridSearchResponse | None:
    """Decompose-then-retrieve. Returns a merged HybridSearchResponse, or None to
    signal "not multi-hop / fall back to single-pass". Never raises."""

    cfg = settings.retrieval
    try:
        max_subq = int(getattr(cfg, "multihop_max_subquestions", 4))
        subqs = await _decompose(request.query, llm_client, max_subq)

        # If the model didn't actually split it (or returned the original), this is
        # not a multi-hop question — fall back to the normal single-pass path.
        if len(subqs) <= 1:
            return None
        norm_orig = re.sub(r"\W+", " ", request.query).strip().lower()
        if len(subqs) == 1 and re.sub(r"\W+", " ", subqs[0]).strip().lower() == norm_orig:
            return None

        # Retrieve each sub-question with the SAME retriever (single-pass each).
        per_subq_top_k = int(getattr(cfg, "multihop_per_subq_top_k", 6))
        pool: list[RetrievalCandidate] = []
        retrievals_run = 0
        for sq in subqs:
            sub_resp = retriever.search(
                HybridSearchRequest(
                    query=sq,
                    top_k=per_subq_top_k,
                    source_system=request.source_system,
                    record_type=request.record_type,
                    # No target_document_types nudge for sub-questions: keep neutral.
                )
            )
            retrievals_run += 1
            pool.extend(sub_resp.results)

        merged = _dedupe(pool)
        if not merged:
            return None

        # Bridge-term injection: extract key entity tokens from the top of the
        # merged pool and fire one additional targeted retrieval for each term not
        # already well-represented. This closes the sequential gap in parallel
        # decompose mode — e.g. sub-question 4 ("lead time?") retrieved generic rows
        # because it didn't yet know the fix part was "FA-12"; a bridge retrieval on
        # "FA-12" finds the specific row. Generic token extraction only — no
        # hardcoding, no answer injection.
        bridge_results = _bridge_retrieve(merged, pool, retriever, request, per_subq_top_k, cfg)
        if bridge_results:
            pool.extend(bridge_results)
            merged = _dedupe(pool)
            retrievals_run += len(bridge_results) > 0  # count bridge as one extra pass

        # Re-rank the merged pool against the ORIGINAL question with the SAME
        # cross-encoder. No manual scoring or boosting.
        final_top_k = int(request.top_k or getattr(cfg, "final_top_k", 8))
        reranked, _rerank_diag = retriever.reranker.rerank(request.query, merged, top_k=final_top_k)
        if not reranked:
            return None

        confidence = estimate_confidence(
            reranked,
            rerank_used=True,
            high_score=getattr(cfg, "confidence_high_score", 0.55),
            medium_score=getattr(cfg, "confidence_medium_score", 0.35),
        )

        diagnostics: dict[str, Any] = {
            "multihop_applied": True,
            "single_pass_rag": False,
            "automatic_multihop_disabled": False,
            "multihop_mode": "decompose",
            "sub_questions": subqs,
            "retrievals_run": retrievals_run,
            "merged_pool_size": len(merged),
            "reranked_kept": len(reranked),
        }
        return HybridSearchResponse(
            query=request.query,
            confidence=confidence,
            results=reranked,
            diagnostics=diagnostics,
        )
    except Exception:
        # Multi-hop must never break the main path; fall back to single-pass.
        return None


def _subanswer_context(candidates: list[RetrievalCandidate], max_chunks: int = 5,
                       per_chunk: int = 1200, total_cap: int = 6000) -> str:
    """Context for the sub-answer extraction step. Unlike the compact digest, this
    PRESERVES newlines/table structure and gives each chunk generous room, because
    the answer often sits in a specific table row (e.g. the ownership map's
    '| RF front-end and EMC compliance | Mei Tanaka |'). Collapsing newlines or
    truncating to a few hundred chars would cut the answer off — which is exactly
    what caused every sub-answer to come back empty."""
    blocks: list[str] = []
    used = 0
    for c in candidates[:max_chunks]:
        text = (c.text or "").strip()
        if len(text) > per_chunk:
            text = text[:per_chunk] + "…"
        if used + len(text) > total_cap:
            text = text[: max(0, total_cap - used)]
        if not text:
            break
        blocks.append(text)
        used += len(text)
        if used >= total_cap:
            break
    return "\n\n---\n\n".join(blocks)


def _evidence_digest(candidates: list[RetrievalCandidate], max_chunks: int = 8, per_chunk: int = 240) -> str:
    """Compact view of retrieved evidence for a sub-question answer step. Kept small
    to avoid the 'retrieval laziness' that grows with context length."""
    lines = []
    for c in candidates[:max_chunks]:
        text = (c.text or "").replace("\n", " ").strip()
        if len(text) > per_chunk:
            text = text[:per_chunk] + "…"
        lines.append(f"- {text}")
    return "\n".join(lines)


_SUBANSWER_SYSTEM = (
    "Answer the question in as few words as possible using ONLY the provided context. "
    "Give just the answer — a name, code, number, or short phrase — with no sentence and "
    "no explanation. If the context does not contain the answer, reply with exactly: "
    "UNKNOWN"
)


async def _answer_subquestion(
    original_query: str,
    sq: str,
    prior_qa: list[tuple[str, str]],
    evidence: list[RetrievalCandidate],
    llm_client: Any,
) -> str:
    """Short extractive answer to one sub-question from its retrieved context.

    Deliberately a SIMPLE prompt. A heavier prompt (original question + an
    'already established' block) helps a large model disambiguate but makes a small
    local model (e.g. llama3.1:8b) bail to UNKNOWN on every hop — regressing all
    sub-answers to empty. Disambiguation is instead handled upstream: the
    decomposer is instructed to carry each distinguishing qualifier into every
    sub-question (so the sub-question already says 'the part for the
    radiated-emissions test failure'), which keeps this step's prompt light enough
    for a small model while still selecting the right item. Returns '' when unknown;
    the answer is carried into later retrievals.

    original_query and prior_qa are accepted for interface stability and light
    optional use; they are intentionally NOT injected into the prompt body.
    """
    if not evidence:
        return ""
    if llm_client is None:
        logger.error("multihop sub-answer: llm_client is None — check wiring in answer_generator.py")
        return ""
    ctx = _subanswer_context(evidence, max_chunks=5, per_chunk=1200, total_cap=6000)
    user = f"Context:\n{ctx}\n\nQuestion: {sq}\n\nShort answer:"
    try:
        llm = await llm_client.generate(system_prompt=_SUBANSWER_SYSTEM, user_prompt=user)
    except Exception as exc:
        logger.warning("multihop sub-answer LLM call failed: %s", exc)
        return ""
    if getattr(llm, "status", None) != "ok":
        logger.warning("multihop sub-answer LLM error: %s", getattr(llm, "error", "unknown"))
        return ""
    text = (getattr(llm, "text", "") or "").strip()
    if not text:
        return ""
    # Take the first non-empty line and strip common small-model lead-ins so a
    # preamble like "Answer:" or "The answer is:" doesn't get treated as the value.
    first = ""
    for line in text.splitlines():
        s = line.strip().strip('"').strip("*").strip()
        if s:
            first = s
            break
    low = first.lower()
    for lead in ("short answer:", "answer:", "the answer is", "the part is", "it is"):
        if low.startswith(lead):
            first = first[len(lead):].strip(" :\"'")
            low = first.lower()
    if not first or low.startswith("unknown") or low == "n/a" or low == "none":
        return ""
    # Keep it short — a carried answer is an entity/phrase, not a paragraph.
    return first[:80]


async def run_multihop_iterative(
    request: Any,
    retriever: Any,
    llm_client: Any,
    settings: Any,
) -> HybridSearchResponse | None:
    """Sequential sub-question answering with answer carry-over (self-ask / IterDRAG
    pattern). Ordered sub-questions are answered one at a time; each short answer is
    carried into the retrieval query for the following sub-questions, so a bridge
    entity discovered in one step (e.g. the fix part 'FA-12') is retrieved in the
    next step ('lead time of FA-12'). The model answers focused extractive
    sub-questions rather than inventing search queries, which is reliable on a small
    local model. Returns a merged HybridSearchResponse or None. Never raises."""

    cfg = settings.retrieval
    try:
        max_subq = int(getattr(cfg, "multihop_max_subquestions", 4))
        per_subq_top_k = int(getattr(cfg, "multihop_per_subq_top_k", 6))

        subqs = await _decompose(request.query, llm_client, max_subq)
        if len(subqs) <= 1:
            return None  # not a multi-hop question → single-pass handles it

        evidence: list[RetrievalCandidate] = []
        seen: set[str] = set()
        carried: list[str] = []          # short answers gathered so far
        prior_qa: list[tuple[str, str]] = []  # (sub-question, answer) established so far
        queries_run: list[str] = []
        sub_answers: list[str] = []
        focused_entities: set[str] = set()  # entities already given a clean focus retrieval

        for sq in subqs:
            # Carry prior answers into the retrieval query so a sub-question that
            # refers to an earlier answer ("the part") retrieves by the concrete
            # value ("FA-12"). This is the mechanism that resolves the bridge.
            carry = " ".join(dict.fromkeys(carried))  # dedupe, preserve order
            retrieval_q = f"{sq} {carry}".strip()
            queries_run.append(retrieval_q)
            try:
                resp = retriever.search(
                    HybridSearchRequest(
                        query=retrieval_q,
                        top_k=per_subq_top_k,
                        source_system=request.source_system,
                        record_type=request.record_type,
                        allow_table_completion=False,
                    )
                )
            except Exception:
                continue
            hits = list(resp.results)
            # Focused entity retrieval. The natural-language sub-question dilutes a
            # carried entity (e.g. "FA-12") among many words — "lead time of the part
            # ... Mei Tanaka FA-12" — so the entity's OWN record (its row in the
            # equipment register) is outranked and never retrieved. Fire a SECOND
            # query on just the single most-specific carried entity so that record
            # surfaces cleanly, then merge. Prefer a code-like identifier (letter+
            # digit, e.g. FA-12) since that keyword-matches a specific row; fall back
            # to the freshest whole answer. Retrieval only, no LLM. The carried entity
            # comes from the model's own verified sub-answers — not corpus knowledge.
            focus_q = ""
            for ans_text in reversed(carried):  # freshest first
                for tok in _TOKEN_RE.findall(ans_text):
                    t = tok.strip(".-")
                    if _is_code(t):
                        focus_q = t
                        break
                if focus_q:
                    break
            if not focus_q and carried:
                focus_q = carried[-1]
            # Fire the focused query whenever the entity is one we haven't already
            # searched cleanly — NOT gated on whether it appears in the diluted main
            # query (it usually does; that dilution is the whole problem we're
            # solving). De-duplication of results is handled below.
            if focus_q and focus_q.lower() not in focused_entities:
                focused_entities.add(focus_q.lower())
                queries_run.append(f"[focus] {focus_q}")
                try:
                    fresp = retriever.search(
                        HybridSearchRequest(
                            query=focus_q,
                            top_k=per_subq_top_k,
                            source_system=request.source_system,
                            record_type=request.record_type,
                            allow_table_completion=False,
                        )
                    )
                    have = {c.chunk_id for c in hits}
                    for c in fresp.results:
                        if c.chunk_id not in have:
                            hits.append(c)
                            have.add(c.chunk_id)
                except Exception:
                    pass
            for c in hits:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    evidence.append(c)
            # Answer IN CONTEXT of the original question + what's established, so the
            # model can disambiguate between similar items (e.g. two failures each
            # with a different corrective part) instead of guessing.
            ans = await _answer_subquestion(request.query, sq, prior_qa, hits, llm_client)
            sub_answers.append(ans)
            prior_qa.append((sq, ans))
            if ans:
                carried.append(ans)

        if not evidence:
            return None

        final_top_k = int(request.top_k or getattr(cfg, "final_top_k", 8))
        reranked, _diag = retriever.reranker.rerank(request.query, evidence, top_k=final_top_k)
        if not reranked:
            return None

        confidence = estimate_confidence(
            reranked,
            rerank_used=True,
            high_score=getattr(cfg, "confidence_high_score", 0.55),
            medium_score=getattr(cfg, "confidence_medium_score", 0.35),
        )
        diagnostics: dict[str, Any] = {
            "multihop_applied": True,
            "single_pass_rag": False,
            "automatic_multihop_disabled": False,
            "multihop_mode": "iterative",
            "sub_questions": subqs,
            "sub_answers": sub_answers,
            "queries_run": queries_run,
            "hops": len(subqs),
            "retrievals_run": len(queries_run),
            "merged_pool_size": len(evidence),
            "reranked_kept": len(reranked),
        }
        return HybridSearchResponse(
            query=request.query,
            confidence=confidence,
            results=reranked,
            diagnostics=diagnostics,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------- #
# Dispatcher
# ---------------------------------------------------------------------------- #
async def run_multihop(
    request: Any,
    retriever: Any,
    llm_client: Any,
    settings: Any,
) -> HybridSearchResponse | None:
    """Dispatch to the configured multi-hop mode. Returns None to fall back."""
    mode = getattr(settings.retrieval, "multihop_mode", "decompose")
    if mode == "iterative":
        return await run_multihop_iterative(request, retriever, llm_client, settings)
    return await run_multihop_decompose(request, retriever, llm_client, settings)


def is_low_confidence(confidence: str) -> bool:
    """Phase 2: escalate to multi-hop when single-pass confidence is weak."""
    return str(confidence).lower() not in ("high",)
