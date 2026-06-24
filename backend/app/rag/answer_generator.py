"""Grounded answer generator.

Purpose
-------
Produces an answer that is grounded in retrieved evidence and carries citations,
or honestly declines when the evidence is too weak.

What it does
------------
Runs retrieval, gates on retrieval confidence, builds the evidence context (merging
prior evidence for follow-up questions), prompts the model, validates the
citations, retries once if needed, and falls back to a clear "no answer" rather
than guessing.

Flow
----
query -> retrieval -> confidence gate -> context build -> grounded prompt ->
model -> map temporary labels to stable citations -> validate -> (retry once) ->
return a supported answer or a safe no-answer response.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.core.config import LLMConfig, Settings
from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient
from backend.app.rag.citation_validator import CitationValidator
from backend.app.rag.context_builder import RAGContextBuilder
from backend.app.rag.models import (
    BuiltRAGContext,
    CitationValidationResult,
    EvidenceChunk,
    RAGAnswerRequest,
    RAGAnswerResponse,
)
from backend.app.rag.prompt_builder import RAGPromptBuilder
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.models import HybridSearchRequest

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}
SHORT_LABEL_PATTERN = re.compile(r"[\[\(]\s*(S\d+)\s*[\]\)]", re.IGNORECASE)


class RAGAnswerGenerator:
    """Generate grounded, cited answers using local retrieval and a local LLM."""

    def __init__(self, settings: Settings, retriever: HybridRetriever | None = None):
        self.settings = settings
        self.retriever = retriever or HybridRetriever(settings)
        self.context_builder = RAGContextBuilder(settings)
        self.prompt_builder = RAGPromptBuilder(settings)
        self.validator = CitationValidator(settings)
        self.llm_client = OpenAICompatibleLLMClient(self._rag_llm_config())

    # ------------------------------------------------------------------ #
    # LLM config (RAG-specific temperature / token budget)
    # ------------------------------------------------------------------ #
    def _rag_llm_config(self) -> LLMConfig:
        base = self.settings.llm
        return LLMConfig(
            provider=base.provider,
            base_url=base.base_url,
            model=base.model,
            temperature=self.settings.rag.answer_temperature,
            max_tokens=self.settings.rag.answer_max_tokens,
            timeout_seconds=base.timeout_seconds,
        )

    async def _classify_query_intent(self, query: str) -> list[str]:
        """Classify which document type(s) the query targets. One small LLM call, fail-open to ['any'] so retrieval is never
        broken by this signal. Gated by config."""

        meta_cfg = getattr(self.settings.retrieval, "metadata_signal", None)
        if not meta_cfg or not getattr(meta_cfg, "enabled", False):
            return ["any"]
        if not getattr(meta_cfg, "query_intent_classification", False):
            return ["any"]
        allowed = list(getattr(meta_cfg, "document_types", []) or [])
        if not allowed:
            return ["any"]
        try:
            from backend.app.retrieval.query_intent import classify_query_intent
            return await classify_query_intent(query, allowed, self.llm_client, enabled=True)
        except Exception:
            return ["any"]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def answer(self, request: RAGAnswerRequest) -> RAGAnswerResponse:
        target_types = await self._classify_query_intent(request.query)
        retrieval = self.retriever.search(
            HybridSearchRequest(
                query=request.query,
                top_k=request.top_k,
                source_system=request.source_system,
                record_type=request.record_type,
                target_document_types=tuple(target_types),
            )
        )

        prior = self._reconstruct_prior_evidence(request.prior_evidence)

        # Confidence gate: refuse early only when retrieval is too weak AND there
        # is no prior evidence to support a follow-up ("draft it", "summarize him").
        too_weak = self._below_min_confidence(retrieval.confidence) or not retrieval.results
        if too_weak and not prior:
            return self._no_answer_response(
                request, retrieval.confidence, retrieval.diagnostics,
                reason="weak_retrieval",
            )

        context = self.context_builder.build(retrieval)
        context = self._merge_prior_evidence(context, prior)

        if not context.evidence:
            return self._no_answer_response(
                request, retrieval.confidence, retrieval.diagnostics,
                reason="no_context",
            )

        return await self._generate(request, retrieval.confidence, retrieval.diagnostics, context)

    # ------------------------------------------------------------------ #
    # Generation + validation + retry
    # ------------------------------------------------------------------ #
    async def _generate(
        self,
        request: RAGAnswerRequest,
        retrieval_confidence: str,
        retrieval_diagnostics: dict[str, Any],
        context: BuiltRAGContext,
    ) -> RAGAnswerResponse:
        attempts = 2 if self.settings.rag.retry_on_invalid_citations else 1
        last_answer = ""
        last_validation: CitationValidationResult | None = None
        last_llm_latency: int | None = None
        model_name = self.settings.llm.model

        for attempt in range(attempts):
            retry = attempt > 0
            prompt = self.prompt_builder.build(
                query=request.query,
                context=context,
                retry=retry,
                answer_mode=request.answer_mode,
                conversation_context=request.conversation_context,
                complete_table_present=bool(retrieval_diagnostics.get("complete_table_present")),
            )
            llm = await self.llm_client.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
            )
            last_llm_latency = llm.latency_ms

            if llm.status != "ok":
                return RAGAnswerResponse(
                    query=request.query,
                    answer="",
                    status="error",
                    confidence=retrieval_confidence,
                    model=model_name,
                    citations=[],
                    validation=CitationValidationResult(False, [], [], [], errors=[llm.error or "LLM error"]),
                    retrieval_diagnostics=retrieval_diagnostics,
                    evidence=context.evidence if request.show_evidence else [],
                    llm_latency_ms=last_llm_latency,
                    error=llm.error,
                    used_retry=retry,
                )

            mapped = self._map_short_to_technical(llm.text, context)
            validation = self.validator.validate(mapped, context)
            last_answer, last_validation = mapped, validation

            if validation.valid:
                no_answer = self.settings.rag.no_answer_message.lower() in mapped.lower()
                return RAGAnswerResponse(
                    query=request.query,
                    answer=mapped,
                    status="no_answer" if no_answer else "supported",
                    confidence=retrieval_confidence,
                    model=model_name,
                    citations=validation.cited_labels if not no_answer else [],
                    validation=validation,
                    retrieval_diagnostics=retrieval_diagnostics,
                    evidence=context.evidence if request.show_evidence else [],
                    llm_latency_ms=last_llm_latency,
                    used_retry=retry,
                )

        # All attempts produced invalid citations. Before falling back to a
        # no-answer, try citation salvage: attribute uncited sentences back to
        # the retrieved evidence by lexical overlap. Honest (only cites chunks
        # that were retrieved and that lexically support the sentence) and a pure
        # last resort. Gated by config.
        if getattr(self.settings.rag, "citation_salvage_enabled", True) and last_answer:
            from backend.app.rag.citation_salvage import salvage_citations

            salvaged = salvage_citations(
                last_answer,
                context.evidence,
                min_overlap=float(getattr(self.settings.rag, "citation_salvage_min_overlap", 0.5)),
            )
            if salvaged != last_answer:
                salvaged_validation = self.validator.validate(salvaged, context)
                if salvaged_validation.valid:
                    no_answer = self.settings.rag.no_answer_message.lower() in salvaged.lower()
                    if not no_answer:
                        diag = dict(retrieval_diagnostics)
                        diag["citation_salvage_used"] = True
                        return RAGAnswerResponse(
                            query=request.query,
                            answer=salvaged,
                            status="supported",
                            confidence=retrieval_confidence,
                            model=model_name,
                            citations=salvaged_validation.cited_labels,
                            validation=salvaged_validation,
                            retrieval_diagnostics=diag,
                            evidence=context.evidence if request.show_evidence else [],
                            llm_latency_ms=last_llm_latency,
                            used_retry=attempts > 1,
                        )

        # Salvage did not help -> safe no-answer fallback.
        return self._no_answer_response(
            request, retrieval_confidence, retrieval_diagnostics,
            reason="invalid_citations",
            validation=last_validation,
            llm_latency_ms=last_llm_latency,
            evidence=context.evidence,
            used_retry=attempts > 1,
        )

    # ------------------------------------------------------------------ #
    # Short-label mapping
    # ------------------------------------------------------------------ #
    # Also strip [E1]-style evidence-ID labels if the model uses them despite instructions.
    _EVIDENCE_ID_PATTERN = re.compile(r"[\[\(]\s*E\d+\s*[\]\)]", re.IGNORECASE)

    @classmethod
    def _map_short_to_technical(cls, answer: str, context: BuiltRAGContext) -> str:
        """Map the model's [S1].. labels back to stable technical citation labels.

        The prompt exposes short labels because small local models cite them
        more reliably; validation and the UI work on the technical labels.

        Also strips any [E1]-style evidence-ID labels the model produces despite
        instructions, so they don't appear as orphaned citations in the final answer.
        """

        if not answer:
            return answer
        mapping = {f"S{i}": ev.citation_label for i, ev in enumerate(context.evidence, start=1)}

        def repl(match: re.Match) -> str:
            key = match.group(1).upper()
            return mapping.get(key, match.group(0))

        mapped = SHORT_LABEL_PATTERN.sub(repl, answer)
        # Strip any [E1]-style labels that weren't converted (model error).
        mapped = cls._EVIDENCE_ID_PATTERN.sub("", mapped)
        return mapped

    # ------------------------------------------------------------------ #
    # Prior evidence (conversation follow-ups)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconstruct_prior_evidence(prior: list[dict[str, Any]] | None) -> list[EvidenceChunk]:
        items: list[EvidenceChunk] = []
        for entry in prior or []:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("citation_label") or "").strip()
            if not label:
                continue
            items.append(
                EvidenceChunk(
                    evidence_id=str(entry.get("evidence_id") or f"P{len(items)+1}"),
                    citation_label=label,
                    chunk_id=str(entry.get("chunk_id") or ""),
                    document_id=str(entry.get("document_id") or ""),
                    source_system=str(entry.get("source_system") or ""),
                    source_file=str(entry.get("source_file") or ""),
                    record_type=str(entry.get("record_type") or ""),
                    title=str(entry.get("title") or ""),
                    text=str(entry.get("text") or entry.get("text_preview") or ""),
                    combined_score=float(entry.get("combined_score") or 0.0),
                    match_reasons=list(entry.get("match_reasons") or ["prior_evidence"]),
                )
            )
        return items

    def _merge_prior_evidence(self, context: BuiltRAGContext, prior: list[EvidenceChunk]) -> BuiltRAGContext:
        if not prior:
            return context
        max_chunks = self.settings.rag.max_context_chunks
        present = {ev.citation_label for ev in context.evidence}
        merged_evidence = list(context.evidence)
        for item in prior:
            if len(merged_evidence) >= max_chunks:
                break
            if item.citation_label in present or not item.text:
                continue
            present.add(item.citation_label)
            merged_evidence.append(item)

        if len(merged_evidence) == len(context.evidence):
            return context

        # Re-index evidence IDs and rebuild context text deterministically.
        reindexed: list[EvidenceChunk] = []
        blocks: list[str] = []
        for index, ev in enumerate(merged_evidence, start=1):
            ev2 = EvidenceChunk(
                evidence_id=f"E{index}", citation_label=ev.citation_label, chunk_id=ev.chunk_id,
                document_id=ev.document_id, source_system=ev.source_system, source_file=ev.source_file,
                record_type=ev.record_type, title=ev.title, text=ev.text,
                combined_score=ev.combined_score, match_reasons=ev.match_reasons,
            )
            reindexed.append(ev2)
            reasons = ", ".join(ev2.match_reasons[:8]) if ev2.match_reasons else "retrieved"
            blocks.append(
                f"Evidence ID: E{index}\nCitation Label: {ev2.citation_label}\nTitle: {ev2.title}\n"
                f"Source System: {ev2.source_system}\nRecord Type: {ev2.record_type}\n"
                f"Source File: {ev2.source_file}\nRetrieval Score: {ev2.combined_score:.4f}\n"
                f"Match Reasons: {reasons}\nEvidence Text:\n{ev2.text}"
            )
        text = "\n\n".join(blocks)
        return BuiltRAGContext(
            query=context.query,
            retrieval_confidence=context.retrieval_confidence,
            evidence=reindexed,
            context_text=text,
            total_chars=len(text),
            truncated=context.truncated,
            diagnostics={**context.diagnostics, "prior_evidence_merged": len(reindexed) - len(context.evidence)},
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _below_min_confidence(self, confidence: str) -> bool:
        floor = self.settings.rag.min_retrieval_confidence
        return CONFIDENCE_ORDER.get(confidence, 0) < CONFIDENCE_ORDER.get(floor, 1)

    def _no_answer_response(
        self,
        request: RAGAnswerRequest,
        retrieval_confidence: str,
        retrieval_diagnostics: dict[str, Any],
        *,
        reason: str,
        validation: CitationValidationResult | None = None,
        llm_latency_ms: int | None = None,
        evidence: list[EvidenceChunk] | None = None,
        used_retry: bool = False,
    ) -> RAGAnswerResponse:
        validation = validation or CitationValidationResult(True, [], [], [], warnings=[f"no_answer:{reason}"])
        return RAGAnswerResponse(
            query=request.query,
            answer=self.settings.rag.no_answer_message,
            status="no_answer",
            confidence=retrieval_confidence,
            model=self.settings.llm.model,
            citations=[],
            validation=validation,
            retrieval_diagnostics={**(retrieval_diagnostics or {}), "no_answer_reason": reason},
            evidence=(evidence or []) if request.show_evidence else [],
            llm_latency_ms=llm_latency_ms,
            used_retry=used_retry,
        )
