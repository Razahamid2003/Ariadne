"""Test the local model connection.

Purpose
-------
Confirms the configured local language model can be reached through the same
adapter the application uses.

Usage
-----
    python scripts/test_llm.py --config config/client.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import load_settings
from backend.app.llm.openai_compatible import OpenAICompatibleLLMClient


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test local LLM connection.")
    parser.add_argument(
        "--config",
        default="config/client.yaml",
        help="Path to config YAML file.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)

    if settings.security.allow_external_calls:
        print("[FAIL] security.allow_external_calls must remain false for local PoC.")
        return 1

    llm = OpenAICompatibleLLMClient(settings.llm)

    response = await llm.generate(
        system_prompt="You are a local test assistant. Reply only with valid JSON.",
        user_prompt='Return exactly this JSON: {"status":"ok","model_ready":true}',
    )

    print(json.dumps(response.__dict__, indent=2, ensure_ascii=False))

    if response.status != "ok":
        print("[FAIL] Model call failed.")
        return 1

    if "model_ready" not in response.text:
        print("[WARN] Model responded, but output did not follow the requested JSON exactly.")

    print("[PASS] Local LLM adapter is working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))