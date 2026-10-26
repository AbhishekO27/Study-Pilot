"""Evaluation cases and the three scores computed from a finished answer."""

from studypilot.evals.cases import EvalCase, load_cases
from studypilot.evals.score import CaseScore, EvalReport, format_report, score_case, summarize

__all__ = [
    "CaseScore",
    "EvalCase",
    "EvalReport",
    "format_report",
    "load_cases",
    "score_case",
    "summarize",
]
