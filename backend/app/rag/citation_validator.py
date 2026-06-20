"""Citation validation.

Purpose
-------
Enforces the core safety rule that every citation in an answer must come from the
retrieved evidence, and that factual answers must cite something.

What it does
------------
Extracts the citations from an answer, checks them against the allowed evidence
labels, recognizes a clean "no answer" as valid, and flags raw chunk-slice
artifacts that should not appear as final text.

Flow
----
The generated answer is parsed for citations, each is checked against the evidence
packet, and the result reports validity plus any errors that should trigger a retry
or fallback.
"""

from __future__ import annotations

import re

from backend.app.core.config import Settings
from backend.app.rag.models import BuiltRAGContext, CitationValidationResult

# Matches technical labels like [DOCS: doc-xyz-chunk-0001]
CITATION_PATTERN = re.compile(r"\[[A-Z0-9_\- ]+\s*:\s*[^\]]+\]")
# Matches orphaned short labels like [S1] that were NOT mapped back to technical
ORPHAN_SHORT_LABEL_PATTERN = re.compile(r"[\[\(]\s*S\d+\s*[\]\)]", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"(?im)^###\s+(Confidence|Missing Information)\s*$")



class CitationValidator:
    """
    Validate generated answer citations against retrieved evidence labels.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate(self, answer: str, context: BuiltRAGContext) -> CitationValidationResult:
        allowed = set(context.citation_labels)
        cited = self.extract_citations(answer)
        cited_set = set(cited)
        errors: list[str] = []
        warnings: list[str] = []

        no_answer = self.settings.rag.no_answer_message.lower() in answer.lower()
        if no_answer and not self._is_clean_no_answer(answer):
            errors.append("Answer mixes unsupported factual content with a no-answer statement.")

        if self.settings.rag.require_citations and not no_answer and not cited:
            errors.append("Answer contains no citations.")

        if not no_answer and self._contains_raw_fragment(answer):
            errors.append("Answer appears to contain a raw broken OCR/chunk fragment.")

        unexpected = sorted(cited_set - allowed)
        if unexpected:
            errors.append("Answer cites labels that were not present in the retrieved evidence.")

        unused = sorted(allowed - cited_set)
        if not cited and context.evidence and not no_answer:
            warnings.append("Evidence was available but the answer did not cite it.")

        if not context.evidence and cited:
            errors.append("Answer cites evidence even though no evidence was provided.")

        return CitationValidationResult(
            valid=not errors,
            cited_labels=cited,
            unused_evidence_labels=unused,
            unexpected_labels=unexpected,
            errors=errors,
            warnings=warnings,
        )


    def _is_clean_no_answer(self, answer: str) -> bool:
        """Return True only when the no-answer message is the actual answer.

        A local model sometimes writes factual content and then appends the
        configured no-answer sentence. That mixed output must be rejected so the
        answer generator can retry or replace it with a clean no-answer.
        """

        if not answer:
            return False
        no_answer = self.settings.rag.no_answer_message.lower().strip()
        if not no_answer:
            return False
        body = self._strip_display_sections(answer).strip()
        lowered = re.sub(r"\s+", " ", body.lower()).strip()
        if not lowered.startswith(no_answer):
            return False
        remainder = lowered[len(no_answer):].strip(" \n\t.-:;")
        if not remainder:
            return True
        return remainder.startswith("validation note")

    @staticmethod
    def _strip_display_sections(answer: str) -> str:
        if not answer:
            return ""
        match = SECTION_PATTERN.search(answer)
        if match:
            return answer[: match.start()].strip()
        return answer.strip()

    @staticmethod
    def _contains_raw_fragment(answer: str) -> bool:
        body = CitationValidator._strip_display_sections(answer or "")
        # Generic signal for a copied broken chunk prefix: a one/two-letter orphan
        # token immediately followed by a function word, e.g. "t by the system".
        # Domain-agnostic: depends only on shape, not on any specific vocabulary.
        if re.search(r"(?:^|[\n.;:!?]\s*)[a-z]{1,2}\s+(?:by|and|or|of|to|in|with)\s+[a-z]", body):
            return True
        return False

    @staticmethod
    def extract_citations(text: str) -> list[str]:
        """
        Return unique citation labels in order of first appearance.
        Includes both technical labels [SOURCE: id] and any orphaned
        short labels [S1] that were not mapped back to technical form.
        """

        seen: set[str] = set()
        citations: list[str] = []

        for match in CITATION_PATTERN.findall(text or ""):
            label = match.strip()
            if label not in seen:
                seen.add(label)
                citations.append(label)

        # Also capture orphaned [S1]-style labels so they surface as
        # "unexpected" in validation and trigger the retry path correctly.
        for match in ORPHAN_SHORT_LABEL_PATTERN.finditer(text or ""):
            label = match.group(0).strip()
            if label not in seen:
                seen.add(label)
                citations.append(label)

        return citations
