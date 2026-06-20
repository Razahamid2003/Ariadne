"""Query-intent classification.

Purpose
-------
Works out which kind of document a question is really looking for (for example a
resume versus a form), so retrieval can prefer that kind of source.

What it does
------------
Asks the local model to map the query to one or more configured document types, and
fails open to "any" when the intent is unclear, so it can only help ranking, never
hide results.

Flow
----
A short prompt lists the allowed types; the model's answer is parsed and validated
against them; the result is used as a soft signal during fusion.
"""

from __future__ import annotations

import re


def build_intent_prompt(query: str, allowed_types: list[str]) -> tuple[str, str]:
    system_prompt = (
        "You map a search query to the kind(s) of document that would answer it. "
        "Choose from the allowed document types. If the query does not clearly "
        "favor any particular type, answer with 'any'. Respond with ONLY a "
        "comma-separated list of type names (or 'any'). No prose."
    )
    user_prompt = (
        "Allowed document types: " + ", ".join(allowed_types) + ", any\n\n"
        f"Query: {query}\n\n"
        "Which document type(s) would answer this query? Comma-separated list only."
    )
    return system_prompt, user_prompt


def parse_intent(raw_text: str, allowed_types: list[str]) -> list[str]:
    """Parse the model output into a validated list of types, or ['any']."""

    if not raw_text:
        return ["any"]
    tokens = re.split(r"[,\n;]+", raw_text.strip().lower())
    allowed_lower = {t.lower(): t for t in allowed_types}
    picked: list[str] = []
    for tok in tokens:
        t = tok.strip().replace(" ", "_")
        if not t:
            continue
        if t == "any":
            return ["any"]
        if t in allowed_lower and allowed_lower[t] not in picked:
            picked.append(allowed_lower[t])
        else:
            for low, original in allowed_lower.items():
                if (low in t or t in low) and original not in picked:
                    picked.append(original)
                    break
    return picked or ["any"]


async def classify_query_intent(
    query: str,
    allowed_types: list[str],
    llm_client,
    *,
    enabled: bool = True,
) -> list[str]:
    """Return the document type(s) the query targets, or ['any'].

    Fail-open: any error or ambiguity yields ['any'] so retrieval is unchanged.
    """

    if not enabled or not allowed_types or not (query or "").strip():
        return ["any"]
    system_prompt, user_prompt = build_intent_prompt(query, allowed_types)
    try:
        response = await llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return ["any"]
    if getattr(response, "status", "error") != "ok":
        return ["any"]
    return parse_intent(getattr(response, "text", "") or "", allowed_types)
