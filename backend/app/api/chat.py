"""Grounded chat and conversation endpoints.

Purpose
-------
Powers the conversational interface: creating chats, listing and opening saved
threads, deleting them, and answering a question with cited evidence.

What it does
------------
Provides CRUD over local chat sessions plus the main ``/chat`` endpoint that runs
a question through retrieval and grounded answer generation, returning the answer,
the evidence actually cited, and diagnostics.

Flow
----
``/chat`` loads the conversation, calls the answer generator, filters the evidence
down to what the answer cited, persists the new turn, and returns a UI-ready
payload. Chat list/get/delete operate on the local chat store.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.api.deps import get_rags_state
from backend.app.api.models import ChatRequest, CreateChatRequest
from backend.app.rag.models import RAGAnswerRequest
from backend.app.rag.diagnostics import single_pass_diagnostics
from backend.app.runtime.app_state import RAGSAppState
from backend.app.services.source_documents import build_source_documents, citation_to_source_map

router = APIRouter(prefix="/api", tags=["chat"])

INDEX_MUTATING_JOBS = {"ingest", "rebuild", "fresh_rebuild"}


def _evidence_used_by_answer(response):
    """Return evidence that was actually cited, not every retrieved chunk.

    This prevents unused retrieval/context rows from leaking into visible source
    lists or future chat context. No-answer responses without citations should not
    carry unrelated evidence forward.
    """

    cited = set(response.citations or [])
    if cited:
        return [item for item in response.evidence if item.citation_label in cited]
    if response.status in {"no_answer", "error"}:
        return []
    return list(response.evidence or [])


@router.get("/chats")
def api_list_chats(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Return saved local chat sessions."""

    return {"chats": state.chat_store.list_chats()}


@router.post("/chats")
def api_create_chat(payload: CreateChatRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Create a new empty local chat."""

    return {"chat": state.chat_store.create_chat(payload.title)}


@router.get("/chats/{chat_id}")
def api_get_chat(chat_id: str, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Return one chat with messages."""

    chat = state.chat_store.get_chat(chat_id, include_messages=True)
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
    return {"chat": chat}


@router.delete("/chats/{chat_id}")
def api_delete_chat(chat_id: str, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Soft-delete a local chat session."""

    deleted = state.chat_store.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found.")
    return {"deleted": True, "chat_id": chat_id}


@router.delete("/chats")
def api_delete_all_chats(state: RAGSAppState = Depends(get_rags_state)) -> dict:
    """Delete every chat session (clear the full archive)."""
    chats = state.chat_store.list_chats(limit=10000)
    deleted = sum(1 for c in chats if state.chat_store.delete_chat(c["chat_id"]))
    return {"deleted": True, "count": deleted}


@router.post("/chat")
async def api_chat(payload: ChatRequest, state: RAGSAppState = Depends(get_rags_state)) -> dict:
    if state.settings.runtime.reject_chat_during_rebuild and (state.index_mutating() or state.job_manager.has_active_job(INDEX_MUTATING_JOBS)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An index-mutating admin job is running. Try again when it finishes.",
        )

    started = time.perf_counter()
    chat = state.chat_store.ensure_chat(payload.chat_id, first_query=payload.query)
    chat_id = chat["chat_id"]
    state.chat_store.rename_chat_if_empty(chat_id, payload.query)

    conversation_context = state.chat_store.conversation_context(chat_id)
    prior_evidence = state.chat_store.prior_evidence(chat_id)
    user_message = state.chat_store.add_message(chat_id, "user", payload.query)

    async with state.concurrency.chat_slot():
        generator = state.get_rag_answer_generator()
        response = await generator.answer(
            RAGAnswerRequest(
                query=payload.query,
                top_k=payload.top_k,
                source_system=payload.source_system,
                record_type=payload.record_type,
                show_evidence=payload.show_evidence,
                answer_mode=payload.answer_mode,
                conversation_context=conversation_context,
                prior_evidence=prior_evidence,
            )
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    # Store only evidence actually used/cited. This keeps later follow-ups grounded
    # without contaminating new turns with unrelated top-k chunks.
    used_evidence = _evidence_used_by_answer(response)
    stored_payload = response.to_dict(preview_chars=5000, include_evidence=True)
    stored_payload["retrieval_diagnostics"] = single_pass_diagnostics(stored_payload.get("retrieval_diagnostics") or {})
    stored_payload["evidence"] = [item.to_dict(preview_chars=5000) for item in used_evidence]
    stored_payload["api_latency_ms"] = latency_ms
    stored_payload["source_documents"] = build_source_documents(used_evidence, preview_chars=1200)
    stored_payload["citation_source_map"] = citation_to_source_map(stored_payload["source_documents"])
    stored_payload["chat_id"] = chat_id

    assistant_message = state.chat_store.add_message(chat_id, "assistant", response.answer, payload=stored_payload)

    result = response.to_dict(
        preview_chars=payload.preview_chars,
        include_evidence=payload.show_evidence,
    )
    result["retrieval_diagnostics"] = single_pass_diagnostics(result.get("retrieval_diagnostics") or {})
    result["api_latency_ms"] = latency_ms
    result["chat_id"] = chat_id
    result["user_message"] = user_message
    result["assistant_message"] = assistant_message
    display_evidence = _evidence_used_by_answer(response)
    if payload.show_evidence and "evidence" in result:
        result["evidence"] = [item.to_dict(preview_chars=payload.preview_chars) for item in display_evidence]
    result["source_documents"] = build_source_documents(display_evidence, preview_chars=payload.preview_chars)
    result["citation_source_map"] = citation_to_source_map(result["source_documents"])

    state.query_logger.log(
        {
            "type": "chat",
            "chat_id": chat_id,
            "query": payload.query,
            "status": response.status,
            "confidence": response.confidence,
            "citations": response.citations,
            "source_documents": [doc.get("display_name") for doc in result["source_documents"]],
            "latency_ms": latency_ms,
            "llm_latency_ms": response.llm_latency_ms,
            "result_count": len(response.evidence),
            "answer_mode": payload.answer_mode,
            "conversation_context_chars": len(conversation_context or ""),
            "prior_evidence_count": len(prior_evidence or []),
        }
    )

    return result
