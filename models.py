"""
Shared Pydantic models used across runners, scorers, and the reporter.

TestCase mirrors the JSONL schema defined in CLAUDE.PROJECT.md §4.
RunResult is the unified output of both the single and batch runners —
it carries everything needed by the reporter and the comparator.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from scorers.deterministic import CheckResult
from scorers.llm_judge import EvalResult


# ── Input models (JSONL test set schema) ─────────────────────────────────────


class TestCase(BaseModel):
    id: str
    input: str
    context: dict
    expected_intent: str
    expected_action: str
    expected_safe: bool
    tags: list[str] = Field(default_factory=list)
    scoring_rubric: str = ""
    notes: str = ""


# ── Output models ─────────────────────────────────────────────────────────────


class RunResult(BaseModel):
    """Full result for one test case, combining pipeline output with all scores.

    Produced by both single_runner.run_single() and batch_runner.run_batch().
    Serialised into reports/runs/<timestamp>.jsonl by the reporter.
    """
    # Source test case
    test_case_id: str
    input: str
    context: dict
    expected_intent: str
    expected_safe: bool
    scoring_rubric: str
    tags: list[str]

    # Pipeline output
    output: str          # PostprocessedResponse.text_for_user
    trace: dict          # PipelineTrace as a plain dict

    # Deterministic check results (always populated)
    deterministic_checks: list[CheckResult]
    deterministic_passed: bool   # True only if ALL four checks passed

    # LLM judge result (None when debug=False, or when deterministic_passed=False)
    judge_result: EvalResult | None = None
