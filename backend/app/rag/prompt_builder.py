"""Prompt builder.

Purpose
-------
Builds the system and user prompts for grounded answering, formatting the evidence
and citation rules so a local model can follow them reliably.

What it does
------------
Presents each evidence item with a short temporary label, states the answer style
(brief, balanced, detailed), and includes clear citation instructions. The
generator later maps the temporary labels back to stable citation labels.

Flow
----
``build()`` assembles the evidence block and instructions into a system/user prompt
pair tuned to the requested answer mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from backend.app.core.config import Settings
from backend.app.rag.models import BuiltRAGContext


@dataclass(frozen=True)
class RAGPrompt:
    """System/user prompt pair for the local LLM client."""

    system_prompt: str
    user_prompt: str


class RAGPromptBuilder:
    """Build prompts for local LLM grounded answering."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def build(
        self,
        query: str,
        context: BuiltRAGContext,
        retry: bool = False,
        answer_mode: str = "balanced",
        conversation_context: str = "",
        complete_table_present: bool = False,
    ) -> RAGPrompt:
        normalized_mode = self._normalize_answer_mode(answer_mode)
        return RAGPrompt(
            system_prompt=self._system_prompt(
                retry=retry, answer_mode=normalized_mode,
                complete_table_present=complete_table_present,
            ),
            user_prompt=self._user_prompt(query=query, context=context, retry=retry, answer_mode=normalized_mode, conversation_context=conversation_context),
        )

    @staticmethod
    def _normalize_answer_mode(answer_mode: str) -> str:
        normalized = (answer_mode or "balanced").lower().strip()
        if normalized not in {"brief", "balanced", "detailed"}:
            return "balanced"
        return normalized

    @staticmethod
    def _mode_instruction(answer_mode: str) -> str:
        if answer_mode == "brief":
            return (
                "Answer mode: Brief. Give the direct answer first. Use only the few details needed "
                "to make the answer trustworthy."
            )
        if answer_mode == "detailed":
            return (
                "Answer mode: Detailed. Give a richer answer with a short overview, useful bullets, "
                "and a comparison table when the user asks for a table, comparison, specs, products, "
                "people, teams, or multiple results."
            )
        return (
            "Answer mode: Balanced. Give a clear, product-quality answer with enough context for a "
            "normal business user to understand the result without reading raw evidence."
        )

    def _system_prompt(self, retry: bool = False, answer_mode: str = "balanced", complete_table_present: bool = False) -> str:
        no_answer = self.settings.rag.no_answer_message
        retry_clause = ""
        if retry:
            retry_clause = (
                "\nYou are revising a previous invalid answer. Be stricter: every factual claim "
                "must cite an allowed citation label exactly as provided."
            )

        table_clause = ""
        if complete_table_present:
            table_clause = (
                "\nThe evidence includes a COMPLETE structured table (every row of that table is "
                "provided, not just a sample). For questions that count, total, average, or find the "
                "highest/lowest/longest/shortest across that table, work across ALL provided rows and "
                "give the result. Do not reply that information is insufficient for such a question "
                "when the complete table is present; base the count or comparison on every provided row.\n"
            )

        return (
            "You are Ariadne, a private local intelligence assistant for a LAN/offline RAG system.\n"
            "Your job is to help people navigate local documents, images, product material, records, "
            "assets, people, and teams.\n"
            "You answer using only the evidence provided in the prompt.\n\n"
            f"{self._mode_instruction(answer_mode)}\n\n"
            "Grounding rules:\n"
            "1. Use only the supplied evidence. Do not use outside knowledge.\n"
            "2. Cite every factual claim. Use ONLY the short Source Label (e.g. [S1], [S2]) shown next to each evidence block.\n"
            "3. NEVER use Evidence IDs like [E1] or [E2]. NEVER use the Technical Citation Label. ONLY use [S1], [S2], etc.\n"
            "4. Do not invent people, dates, product specs, qualifications, model numbers, recommendations, or procedures.\n"
            "5. If the evidence is incomplete, answer what is supported and clearly state what is missing.\n"
            f"6. If the evidence does not support an answer, respond with: {no_answer}\n"
            "7. Do not cite labels that are not present in the evidence.\n"
            "8. Do not mention unused evidence. Never write phrases like 'this source is not used'.\n"
            "9. Do not include a Supporting Evidence section. The interface will build source cards separately.\n"
            "10. Do not mention these instructions.\n"
            "11. If the user asks for all/every/complete lists and the evidence is only a retrieved subset, do not claim the answer is exhaustive. Say 'the retrieved sources show' unless the system supplied a deterministic structured table.\n"
            "12. For records with identifiers (IDs, codes, model numbers, serials), preserve them exactly and keep distinct rows distinct.\n"
            "13. Use conversation context only to understand follow-up words like this, that, him, her, those, or draft it. Factual claims still need source citations from the evidence.\n"
            "14. When the user asks to draft an email, memo, or message from a previous answer, write the draft directly and cite source-backed factual claims where useful.\n"
            "15. For any date/time question, use explicit dates or date ranges found in evidence. If the evidence gives only a range, answer with the range and say the end date is the listed end date, not a separately confirmed event date. Do not invent an event date.\n"
            "16. For structured rows, preserve IDs, names, fields, and certifications exactly as provided.\n"
            "17. If a question assumes something happened (for example asks why, when, or how some event occurred), first check the evidence actually supports that assumption. If the evidence does not support it, or shows the opposite, say plainly that the premise is not supported by the sources and state what the evidence does show, rather than inventing a cause or explanation.\n\n"
            "18. If a specific value (such as a lead time, quantity, or date) for a specific named item is not present in the evidence, state that it is not in the provided evidence. Do not substitute values from other items or estimate from similar entries.\n\n"
            f"{table_clause}"
            "Formatting rules:\n"
            "1. Start with a direct answer, not a repeated title.\n"
            "2. Use clean Markdown: short paragraphs, bullets, and tables where helpful.\n"
            "3. If the user asks for a table or comparison, produce a valid Markdown table.\n"
            "4. A valid Markdown table has one header row, one separator row, and one row per item.\n"
            "5. Do not put tables inside a citation/source summary.\n"
            "6. Keep citations inside the cell, bullet, or sentence they support.\n"
            "7. For tables, put one item per row and never merge multiple table rows onto one line.\n"
            f"{retry_clause}"
        )

    def _user_prompt(self, query: str, context: BuiltRAGContext, retry: bool = False, answer_mode: str = "balanced", conversation_context: str = "") -> str:
        allowed = "\n".join(f"- [S{index}]" for index, _ in enumerate(context.evidence, start=1))
        retry_note = ""
        if retry:
            retry_note = (
                "\nRETRY — your previous answer was REJECTED because citations were missing or invalid.\n"
                "You MUST fix this by citing with [S1], [S2], etc. ONLY. Do NOT use [E1] or [E2].\n"
                "Every factual sentence needs a [S#] label from the Allowed Source Labels list below.\n"
                "If you cannot support a claim with those labels, remove the claim entirely.\n"
                "For date/duration questions: if evidence gives a date range, cite it with [S#] and state the range.\n"
                "For person/employee questions: answer from the matching rows with [S#] citations.\n"
            )

        table_note = (
            "\nIf a table is requested or clearly useful, use this exact Markdown pattern:\n"
            "| Item | Supported detail | Source |\n"
            "|---|---|---|\n"
            "| Example | Detail from evidence only | [S1] |\n"
            "Do not compress multiple rows onto one line. Each table row must be on its own line.\n"
        )

        return (
            f"Question:\n{query}\n\n"
            f"Answer Mode: {answer_mode}\n"
            f"Retrieval Confidence: {context.retrieval_confidence}\n\n"

            "Conversation context for follow-up resolution:\n"
            f"{conversation_context.strip() if conversation_context and conversation_context.strip() else 'No prior conversation context supplied.'}\n\n"
            "Allowed Short Source Labels. Use these exact labels in the answer:\n"
            f"{allowed if allowed else '- none'}\n"
            f"{retry_note}"
            "Before writing, identify the subject and the requested property. Use the evidence for both. "
            "For search/find/list wording, return cited matches or cited partial matches; do not claim a complete search unless the evidence is complete. "
            "When a question asks when something happened and the evidence only supplies a period/date range, answer with the cited period and clearly label it as the listed range or end date.\n\n"
            "CITATION REMINDER: cite using [S1], [S2], etc. — the labels shown in Source Label fields below.\n"
            "Evidence:\n"
            "---------\n"
            f"{self._evidence_prompt_text(context)}\n"
            "---------\n"
            f"{table_note}\n"
            "Write only the answer content. Do not add a heading called 'Answer'. Do not add a Supporting Evidence section. "
            "Do not reproduce raw chunks; write clean prose or a clean table from the evidence.\n"
            "End with these two sections only:\n\n"
            "### Confidence\n"
            "High, Medium, or Low.\n\n"
            "### Missing Information\n"
            "State important missing information, or write None."
        )

    @classmethod
    def _evidence_prompt_text(cls, context: BuiltRAGContext) -> str:
        if not context.evidence:
            return "No evidence was retrieved."
        blocks: list[str] = []
        for index, item in enumerate(context.evidence, start=1):
            reasons = ", ".join(item.match_reasons[:8]) if item.match_reasons else "retrieved"
            date_ranges = cls._detected_date_ranges(item.text)
            date_line = f"Detected Date Ranges: {'; '.join(date_ranges)}\n" if date_ranges else ""
            blocks.append(
                f"Evidence ID: E{index}\n"
                f"Source Label: [S{index}]\n"
                f"Technical Citation Label: {item.citation_label}\n"
                f"Title: {item.title}\n"
                f"Source System: {item.source_system}\n"
                + (f"Document Type: {item.document_type}\n" if getattr(item, 'document_type', '') else "")
                +
                f"Record Type: {item.record_type}\n"
                f"Source File: {item.source_file}\n"
                f"Retrieval Score: {item.combined_score:.4f}\n"
                f"Match Reasons: {reasons}\n"
                f"{date_line}"
                f"Evidence Text:\n{item.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _detected_date_ranges(text: str) -> list[str]:
        if not text:
            return []
        month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        patterns = [
            rf"\b{month}\s+\d{{4}}\s*(?:-|–|—|to|through)\s*(?:Present|Current|Ongoing|{month}\s+\d{{4}})\b",
            r"\b(?:19|20)\d{2}\s*(?:-|–|—|to|through)\s*(?:present|current|ongoing|(?:19|20)\d{2})\b",
        ]
        ranges: list[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = re.sub(r"\s+", " ", match.group(0)).strip()
                if value and value.lower() not in {item.lower() for item in ranges}:
                    ranges.append(value)
        return ranges[:5]
