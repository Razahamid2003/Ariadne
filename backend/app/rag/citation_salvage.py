"""Citation salvage.

Purpose
-------
Recovers a correct answer when the model produced grounded text but forgot to
attach its citation markers, so a good answer is not thrown away on a formatting
slip.

What it does
------------
Matches each uncited answer sentence against the retrieved evidence by word overlap
and attaches the best-matching source's citation, only ever citing evidence that
was actually retrieved.

Flow
----
As a last step before giving up, each uncited sentence is compared to the evidence;
if its overlap with a chunk is high enough, that chunk's citation is appended.
Unsupported sentences are left uncited.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Anything that already looks like a citation label, e.g. [DOCS: ...] or [S1].
_HAS_CITATION = re.compile(r"\[[^\]]+\]")
# Lines that begin the trailing meta sections we must not annotate.
_SECTION_HEADER = re.compile(r"^\s*#{1,6}\s*(Confidence|Missing Information)\b", re.IGNORECASE)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "as", "at", "by", "that", "this", "it",
    "from", "has", "have", "had", "we", "can", "his", "her", "their", "they",
    "there", "based", "provided", "evidence", "source", "sources", "subject",
    "requested", "property", "following", "identify", "details", "information",
}


def _content_tokens(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD.findall(text or "")
        if len(w) > 2 and w.lower() not in _STOPWORDS
    }


def _overlap_coefficient(sentence_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of the sentence's content words that appear in the chunk."""

    if not sentence_tokens:
        return 0.0
    return len(sentence_tokens & chunk_tokens) / len(sentence_tokens)


def salvage_citations(answer: str, evidence, min_overlap: float = 0.5) -> str:
    """Append citation labels to uncited body sentences that match evidence.

    Args:
        answer: the model's answer (already mapped to technical labels).
        evidence: iterable of EvidenceChunk (each has .text and .citation_label).
        min_overlap: minimum content-word overlap coefficient to attribute.

    Returns the answer with citation labels appended to supported, previously
    uncited sentences. Leaves everything else untouched.
    """

    if not answer or not evidence:
        return answer

    chunk_profiles = [(ev.citation_label, _content_tokens(ev.text)) for ev in evidence]
    chunk_profiles = [(label, toks) for label, toks in chunk_profiles if toks]
    if not chunk_profiles:
        return answer

    out_lines: list[str] = []
    in_meta_section = False

    for line in answer.split("\n"):
        if _SECTION_HEADER.match(line):
            in_meta_section = True
        if in_meta_section or not line.strip():
            out_lines.append(line)
            continue

        # Process sentence by sentence within the line.
        sentences = _SENTENCE_SPLIT.split(line)
        rebuilt: list[str] = []
        for sentence in sentences:
            if not sentence.strip() or _HAS_CITATION.search(sentence):
                rebuilt.append(sentence)
                continue
            s_tokens = _content_tokens(sentence)
            best_label, best_score = None, 0.0
            for label, toks in chunk_profiles:
                score = _overlap_coefficient(s_tokens, toks)
                if score > best_score:
                    best_label, best_score = label, score
            if best_label and best_score >= min_overlap:
                rebuilt.append(sentence.rstrip() + f" {best_label}")
            else:
                rebuilt.append(sentence)
        out_lines.append(" ".join(rebuilt))

    return "\n".join(out_lines)
