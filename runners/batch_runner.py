"""
Batch runner — evaluates a full JSONL test set against the Leeg pipeline.

Workflow:
  1. Load test cases from a JSONL file (one TestCase per line).
  2. Split into chunks of ≤ 50 (the /api/pipeline/run-batch API limit).
  3. POST each chunk to /api/pipeline/run-batch and collect PipelineTrace responses.
  4. Run all four deterministic checks on every result.
  5. Run the LLM judge ONLY on results where all deterministic checks passed.
  6. Return the full list of RunResult objects.

The deterministic gate (step 4→5) is the critical efficiency mechanism:
obviously broken outputs do not consume judge API tokens.
"""

import json
from pathlib import Path

import httpx
import structlog
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from models import RunResult, TestCase
from scorers.deterministic import run_all_checks
from scorers.llm_judge import LLMJudge

log = structlog.get_logger(__name__)

_BATCH_SIZE = 50   # API hard limit per request


def load_test_cases(path: str) -> list[TestCase]:
    """Load and validate test cases from a JSONL file."""
    cases = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Test set not found: {path}")
    with p.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(TestCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid test case at {path}:{lineno} — {exc}") from exc
    return cases


async def run_batch(
    test_set_path: str,
    client: httpx.AsyncClient,
    judge: LLMJudge | None = None,
) -> list[RunResult]:
    """Evaluate all test cases in a JSONL file against the Leeg pipeline.

    Args:
        test_set_path: Path to the JSONL test set file.
        client:        Authenticated httpx.AsyncClient pointed at the Leeg API.
        judge:         LLMJudge instance. When provided, cases that pass all
                       deterministic checks are sent to the judge for scoring.
                       When None, judge_result is always None.

    Returns:
        List of RunResult objects in the same order as the input test cases.
    """
    cases = load_test_cases(test_set_path)
    total = len(cases)
    log.info("batch_runner.start", test_set=test_set_path, total_cases=total)

    results: list[RunResult] = []

    chunks = [cases[i : i + _BATCH_SIZE] for i in range(0, total, _BATCH_SIZE)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        pipeline_task = progress.add_task("[cyan]Running pipeline…", total=total)
        judge_task = progress.add_task("[yellow]Judging…", total=total)

        # ── Pipeline calls (chunked) ──────────────────────────────────────────
        api_responses: list[dict] = []
        for chunk in chunks:
            payload = {
                "inputs": [
                    {"input": c.input, "context": c.context} for c in chunk
                ]
            }
            try:
                resp = await client.post("/api/pipeline/run-batch", json=payload)
                resp.raise_for_status()
                batch_data = resp.json()
            except httpx.HTTPStatusError as exc:
                log.error(
                    "batch_runner.http_error",
                    status=exc.response.status_code,
                    chunk_size=len(chunk),
                )
                raise
            except httpx.ConnectError as exc:
                log.error("batch_runner.connect_error", error=str(exc))
                raise

            api_responses.extend(batch_data["results"])
            progress.advance(pipeline_task, len(chunk))

        # ── Deterministic checks ──────────────────────────────────────────────
        for case, api_resp in zip(cases, api_responses):
            output: str = api_resp["response"]["text_for_user"]
            trace: dict = api_resp["trace"]
            channel: str = case.context.get("channel", "sms")

            det_checks = run_all_checks(
                output=output,
                channel=channel,
                trace=trace,
                expected_safe=case.expected_safe,
                expected_intent=case.expected_intent,
            )
            det_passed = all(c.passed for c in det_checks)

            results.append(RunResult(
                test_case_id=case.id,
                input=case.input,
                context=case.context,
                expected_intent=case.expected_intent,
                expected_safe=case.expected_safe,
                scoring_rubric=case.scoring_rubric,
                tags=case.tags,
                output=output,
                trace=trace,
                deterministic_checks=det_checks,
                deterministic_passed=det_passed,
                judge_result=None,
            ))

        # ── LLM judge (only on deterministic-passing results) ─────────────────
        if judge is not None:
            for result in results:
                if result.deterministic_passed:
                    result.judge_result = await judge.score(
                        pipeline_input=result.input,
                        pipeline_output=result.output,
                        pipeline_trace=result.trace,
                        scoring_rubric=result.scoring_rubric,
                        context=result.context,
                    )
                else:
                    log.debug(
                        "batch_runner.judge_skipped",
                        case_id=result.test_case_id,
                        failed_checks=[c.check for c in result.deterministic_checks if not c.passed],
                    )
                progress.advance(judge_task, 1)
        else:
            progress.advance(judge_task, total)

    det_pass_count = sum(1 for r in results if r.deterministic_passed)
    judge_count = sum(1 for r in results if r.judge_result is not None)
    log.info(
        "batch_runner.done",
        total=total,
        det_passed=det_pass_count,
        judge_scored=judge_count,
        judge_skipped=total - judge_count,
    )

    return results
