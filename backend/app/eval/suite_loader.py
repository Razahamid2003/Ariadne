"""Evaluation suite loader.

Purpose
-------
Loads a test suite from JSON or YAML without requiring a hard YAML dependency,
raising a clear error if the file is malformed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SuiteLoadError(ValueError):
    """Raised when a suite file cannot be parsed or has the wrong shape."""


def load_suite(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML suite.

    YAML is supported when PyYAML is installed. To keep the PoC dependency-light,
    JSON is always supported. The bundled .yaml file is written as JSON-compatible
    YAML, so it works even without PyYAML.
    """

    suite_path = Path(path)
    if not suite_path.exists():
        raise SuiteLoadError(f"Suite file not found: {suite_path}")
    raw = suite_path.read_text(encoding="utf-8")
    data: Any

    if suite_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(raw)
        except ModuleNotFoundError:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SuiteLoadError(
                    f"{suite_path} is YAML, but PyYAML is not installed and the file is not JSON-compatible YAML. "
                    "Install pyyaml or save the suite as strict JSON."
                ) from exc
        except Exception as exc:  # pragma: no cover - parser-specific
            raise SuiteLoadError(f"Could not parse suite {suite_path}: {exc}") from exc
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SuiteLoadError(f"Could not parse JSON suite {suite_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise SuiteLoadError("Suite root must be an object/dict.")
    tests = data.get("tests")
    if not isinstance(tests, list) or not tests:
        raise SuiteLoadError("Suite must include a non-empty tests list.")

    for index, test in enumerate(tests, start=1):
        if not isinstance(test, dict):
            raise SuiteLoadError(f"Test #{index} must be an object/dict.")
        if not str(test.get("id", "")).strip():
            raise SuiteLoadError(f"Test #{index} is missing id.")
        if not str(test.get("query", "")).strip():
            raise SuiteLoadError(f"Test {test.get('id')} is missing query.")

    data.setdefault("suite_name", suite_path.stem)
    data.setdefault("defaults", {})
    return data
