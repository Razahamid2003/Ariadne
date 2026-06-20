"""Offline evaluation helpers.

Purpose
-------
Contains the checks and utilities used to evaluate retrieval and answer quality
against local test suites, without any cloud services.
"""

from backend.app.eval.suite_loader import load_suite
from backend.app.eval.answer_checks import check_answer_response
from backend.app.eval.retrieval_checks import check_search_response
from backend.app.eval.report_writer import write_markdown_report

__all__ = [
    "load_suite",
    "check_answer_response",
    "check_search_response",
    "write_markdown_report",
]
