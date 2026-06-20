"""Run all model-free tests.

Purpose
-------
Runs the full suite of tests that need no model downloads, giving a fast overall
pass/fail for the backend logic.
"""

import subprocess, sys
from pathlib import Path

SUITES = ["test_fusion_and_retriever", "test_answer_generator", "test_chunker", "test_airgap", "test_citation_fixes", "test_metadata_classification", "test_citation_salvage"]
here = Path(__file__).resolve().parent
failed = 0
for suite in SUITES:
    print(f"\n{'='*60}\n{suite}\n{'='*60}")
    r = subprocess.run([sys.executable, str(here / f"{suite}.py")])
    failed += (r.returncode != 0)
print(f"\n{'#'*60}")
print("ALL SUITES PASSED" if not failed else f"{failed} SUITE(S) FAILED")
sys.exit(1 if failed else 0)
