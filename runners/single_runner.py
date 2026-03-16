"""
Single runner — submits one input to /api/pipeline/run and scores the result.

In normal mode (debug=False) only deterministic checks are run; the LLM judge
is not invoked. In debug mode (debug=True) the full trace and judge reasoning
are printed to stdout with rich for interactive investigation.

Usage:
    client = httpx.AsyncClient(base_url=..., headers={"Authorization": ...})
    result = await run_single(test_case, client, judge=judge, debug=True)
"""

import json

import httpx
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from models import RunResult, TestCase
from scorers.deterministic import run_all_checks
from scorers.llm_judge import LLMJudge

log = structlog.get_logger(__name__)
console = Console()


async def run_single(
    test_case: TestCase,
    client: httpx.AsyncClient,
    judge: LLMJudge | None = None,
    debug: bool = False,
) -> RunResult:
    """Run one test case through the pipeline and score the response.

    Args:
        test_case: The TestCase to evaluate (from a JSONL test set or ad-hoc).
        client:    Authenticated httpx.AsyncClient pointed at the Leeg API.
        judge:     LLMJudge instance. Required for judge scoring; if None,
                   judge_result will always be None.
        debug:     If True, print the full trace and judge reasoning to stdout.

    Returns:
        RunResult with deterministic checks and (if debug=True) judge_result.
    """
    log.info("single_runner.start", case_id=test_case.id, input=test_case.input[:60])

    # ── Call the pipeline ─────────────────────────────────────────────────────
    try:
        resp = await client.post(
            "/api/pipeline/run",
            json={"input": test_case.input, "context": test_case.context},
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        log.error("single_runner.http_error", status=exc.response.status_code, case_id=test_case.id)
        raise
    except httpx.ConnectError as exc:
        log.error("single_runner.connect_error", error=str(exc))
        raise

    output: str = data["response"]["text_for_user"]
    trace: dict = data["trace"]
    channel: str = test_case.context.get("channel", "sms")

    # ── Deterministic checks ──────────────────────────────────────────────────
    det_checks = run_all_checks(
        output=output,
        channel=channel,
        trace=trace,
        expected_safe=test_case.expected_safe,
        expected_intent=test_case.expected_intent,
    )
    det_passed = all(c.passed for c in det_checks)

    # ── LLM judge (debug mode only) ───────────────────────────────────────────
    judge_result = None
    if debug and judge is not None and det_passed:
        judge_result = await judge.score(
            pipeline_input=test_case.input,
            pipeline_output=output,
            pipeline_trace=trace,
            scoring_rubric=test_case.scoring_rubric,
            context=test_case.context,
        )
    elif debug and not det_passed:
        log.info(
            "single_runner.judge_skipped",
            reason="deterministic_checks_failed",
            case_id=test_case.id,
        )

    result = RunResult(
        test_case_id=test_case.id,
        input=test_case.input,
        context=test_case.context,
        expected_intent=test_case.expected_intent,
        expected_safe=test_case.expected_safe,
        scoring_rubric=test_case.scoring_rubric,
        tags=test_case.tags,
        output=output,
        trace=trace,
        deterministic_checks=det_checks,
        deterministic_passed=det_passed,
        judge_result=judge_result,
    )

    if debug:
        _print_debug(result)

    log.info(
        "single_runner.done",
        case_id=test_case.id,
        det_passed=det_passed,
        judge_passed=judge_result.passed if judge_result else None,
    )

    return result


# ── Rich debug output ─────────────────────────────────────────────────────────


def _print_debug(result: RunResult) -> None:
    """Print a full debug view of one RunResult using rich."""
    console.rule(f"[bold cyan]{result.test_case_id}[/bold cyan]")

    # Input / output
    console.print(Panel(result.input, title="Input", border_style="blue"))
    console.print(Panel(result.output, title="Pipeline Output", border_style="green"))

    # Trace
    trace_table = Table(title="Pipeline Trace", show_header=True, header_style="bold magenta")
    trace_table.add_column("Field", style="dim")
    trace_table.add_column("Value")

    guard = result.trace.get("guard_result", {})
    trace_table.add_row("intent", guard.get("intent", "—"))
    trace_table.add_row("intent_confidence", f"{guard.get('confidence', 0):.2f}")
    trace_table.add_row("guard_fired", str(not guard.get("is_safe", True)))
    trace_table.add_row("guard_reason", guard.get("reason", "—"))
    trace_table.add_row("rag_chunks_retrieved", str(result.trace.get("rag_chunks_retrieved", 0)))
    trace_table.add_row("rag_chunks_used", str(result.trace.get("rag_chunks_after_rerank", 0)))
    trace_table.add_row("llm_tokens_prompt", str(result.trace.get("llm_tokens_prompt", 0)))
    trace_table.add_row("llm_tokens_completion", str(result.trace.get("llm_tokens_completion", 0)))

    timings = result.trace.get("stage_timings", {})
    for stage, secs in timings.items():
        trace_table.add_row(f"timing.{stage}", f"{secs * 1000:.0f} ms")

    console.print(trace_table)

    # Deterministic checks
    check_table = Table(title="Deterministic Checks", show_header=True, header_style="bold yellow")
    check_table.add_column("Check", style="dim")
    check_table.add_column("Result")
    check_table.add_column("Detail")

    for c in result.deterministic_checks:
        status_str = "[green]PASS[/green]" if c.passed else "[red]FAIL[/red]"
        check_table.add_row(c.check, status_str, c.detail)

    console.print(check_table)

    # Judge result
    if result.judge_result is not None:
        jr = result.judge_result
        judge_table = Table(title="LLM Judge Scores", show_header=True, header_style="bold blue")
        judge_table.add_column("Dimension")
        judge_table.add_column("Score")
        judge_table.add_column("Pass?")

        for dim in ("accuracy", "groundedness", "safety", "format"):
            score = getattr(jr, dim)
            passed_str = "[green]✓[/green]" if score >= 0.7 else "[red]✗[/red]"
            judge_table.add_row(dim, f"{score:.2f}", passed_str)

        console.print(judge_table)
        overall = "[green]OVERALL PASS[/green]" if jr.passed else "[red]OVERALL FAIL[/red]"
        console.print(f"\n{overall}  (model: {jr.judge_model})")
        console.print(Panel(jr.reasoning, title="Judge Reasoning", border_style="dim"))
    else:
        console.print("[dim]Judge not invoked (debug=False or deterministic checks failed)[/dim]")

    console.rule()
