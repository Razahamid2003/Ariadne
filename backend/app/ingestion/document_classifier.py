"""Document-type classification at ingestion.

Purpose
-------
Labels each document with a type (for example resume, form, report) so retrieval
can later prefer the right kind of source for a question.

What it does
------------
Uses the local language model to read a sample of each document and assign one of
the configured types. The set of types and their descriptions live in
configuration, not in code, and any failure falls back to a safe default so
ingestion is never blocked.

Flow
----
A prompt is built from the configured fields, the model returns a value, and the
result is validated against the allowed set. The chosen type is stored on the
document and copied onto its chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetadataField:
    """One config-defined metadata field."""

    name: str
    description: str = ""
    allowed_values: tuple[str, ...] = ()
    value_descriptions: dict[str, str] | None = None
    fallback: str = "other"

    @classmethod
    def from_config(cls, raw: dict[str, Any]) -> "MetadataField":
        return cls(
            name=str(raw["name"]),
            description=str(raw.get("description", "")),
            allowed_values=tuple(raw.get("allowed_values", []) or []),
            value_descriptions=dict(raw.get("value_descriptions", {}) or {}),
            fallback=str(raw.get("fallback", "other")),
        )


def build_classification_prompt(text_sample: str, fields: list[MetadataField]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for extracting all fields at once."""

    field_specs: list[str] = []
    for f in fields:
        spec = f'- "{f.name}": {f.description}'.rstrip()
        if f.allowed_values:
            spec += "\n  Allowed values (choose exactly one):"
            for v in f.allowed_values:
                vd = (f.value_descriptions or {}).get(v, "")
                spec += f"\n    * {v}" + (f" — {vd}" if vd else "")
        field_specs.append(spec)

    system_prompt = (
        "You classify documents. You are given the opening text of one document "
        "and a set of metadata fields to fill in. Judge by the document's "
        "structure and purpose, not by isolated words. Respond with ONLY a JSON "
        "object mapping each field name to its value. For fields with allowed "
        "values, you MUST choose exactly one of them. No prose, no code fences."
    )
    user_prompt = (
        "Metadata fields to extract:\n"
        + "\n".join(field_specs)
        + "\n\nDocument text (opening excerpt):\n"
        + "-----\n"
        + text_sample
        + "\n-----\n\n"
        + "Return only the JSON object."
    )
    return system_prompt, user_prompt


def _coerce_value(field: MetadataField, value: Any) -> str:
    """Validate a single extracted value against the field's allowed set."""

    text = str(value).strip().lower().replace(" ", "_") if value is not None else ""
    if not field.allowed_values:
        return str(value).strip() if value is not None else field.fallback
    allowed_lower = {v.lower(): v for v in field.allowed_values}
    if text in allowed_lower:
        return allowed_lower[text]
    # Tolerant match: the model sometimes returns a close variant.
    for low, original in allowed_lower.items():
        if low in text or text in low:
            return original
    return field.fallback


def parse_classification(raw_text: str, fields: list[MetadataField]) -> dict[str, str]:
    """Parse the model's JSON output into a validated {field: value} dict.

    Always returns a value for every field; unparseable -> per-field fallback.
    """

    result: dict[str, str] = {f.name: f.fallback for f in fields}
    if not raw_text:
        return result

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return result
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return result
    if not isinstance(data, dict):
        return result

    lower_keys = {str(k).lower(): k for k in data}
    for field in fields:
        key = lower_keys.get(field.name.lower())
        if key is not None:
            result[field.name] = _coerce_value(field, data[key])
    return result


async def classify_document(
    text_sample: str,
    fields: list[MetadataField],
    llm_client,
    *,
    enabled: bool = True,
) -> dict[str, str]:
    """Classify one document. Graceful: any failure -> per-field fallback.

    Returns a {field_name: value} dict suitable for merging into metadata.
    """

    if not enabled or not fields or not (text_sample or "").strip():
        return {f.name: f.fallback for f in fields}

    system_prompt, user_prompt = build_classification_prompt(text_sample, fields)
    try:
        response = await llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    except Exception:
        return {f.name: f.fallback for f in fields}
    if getattr(response, "status", "error") != "ok":
        return {f.name: f.fallback for f in fields}
    return parse_classification(getattr(response, "text", "") or "", fields)
