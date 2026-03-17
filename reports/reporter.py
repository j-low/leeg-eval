"""
Reporter — aggregates RunResult lists into a structured Report, renders a
rich terminal table, and persists results to a timestamped JSONL file.

Save format (reports/runs/<run_id>.jsonl):
  Line 1:   {"_type": "report_summary", ...Report fields...}
  Lines 2+: {"_type": "run_result", ...RunResult fields...}

This layout lets the comparator (Phase 12) read just the first line of each
file to get aggregate metrics without loading all individual results.
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import structlog
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from models import RunResult

log = structlog.get_logger(__name__)
console = Console()

# ── Aggregate models ──────────────────────────────────────────────────────────


class DimensionStats(BaseModel):
    mean_score: float
    pass_rate: float
    pass_count: int
    total_count: int       # cases where the judge was called
    failing_cases: list[str]  # test_case_ids of failing cases (up to 5)


class TagStats(BaseModel):
    pass_rate: float
    pass_count: int
    total_count: int


class Report(BaseModel):
    run_id: str            # YYYY-MM-DDTHH-MM-SS timestamp
    test_set_path: str
    total_cases: int       # all cases in the test set
    evaluated_cases: int   # cases where the LLM judge was called
    skipped_cases: int     # cases where deterministic checks failed (judge not called)

    accuracy: DimensionStats
    groundedness: DimensionStats
    safety: DimensionStats
    format: DimensionStats

    overall_pass_rate: float   # fraction of judged cases that passed all four dimensions
    overall_pass_count: int

    by_tag: dict[str, TagStats]
    timestamp: str             # ISO 8601


# ── Core aggregation ──────────────────────────────────────────────────────────


def generate_report(results: list[RunResult], test_set_path: str = "") -> Report:
    """Aggregate a list of RunResults into a Report.

    Dimension scores and pass rates are computed only over cases where the LLM
    judge ran (i.e. all deterministic checks passed). Cases that failed a
    deterministic check are counted as overall failures.

    Tag pass rates use the full result set: a case passes for a tag if it
    passed all deterministic checks AND (if judged) the judge result passed.
    """
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%S")
    timestamp = now.isoformat()

    total = len(results)
    judged = [r for r in results if r.judge_result is not None]
    skipped = total - len(judged)

    def _dim_stats(dimension: str) -> DimensionStats:
        if not judged:
            return DimensionStats(
                mean_score=0.0, pass_rate=0.0, pass_count=0,
                total_count=0, failing_cases=[],
            )
        scores = [getattr(r.judge_result, dimension) for r in judged]
        passing = [r for r in judged if getattr(r.judge_result, dimension) >= 0.7]
        failing = [r for r in judged if getattr(r.judge_result, dimension) < 0.7]
        return DimensionStats(
            mean_score=round(sum(scores) / len(scores), 3),
            pass_rate=round(len(passing) / len(judged), 3),
            pass_count=len(passing),
            total_count=len(judged),
            failing_cases=[r.test_case_id for r in failing[:5]],
        )

    # Overall pass: deterministic AND judge passed (or deterministic-only for skipped)
    def _overall_passed(r: RunResult) -> bool:
        if not r.deterministic_passed:
            return False
        if r.judge_result is not None:
            return r.judge_result.passed
        return True   # deterministic passed, no judge = counted as pass

    overall_passing = [r for r in results if _overall_passed(r)]

    # Tag aggregation
    tag_buckets: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        for tag in r.tags:
            tag_buckets[tag].append(r)

    by_tag: dict[str, TagStats] = {}
    for tag, tag_results in tag_buckets.items():
        tag_passing = [r for r in tag_results if _overall_passed(r)]
        by_tag[tag] = TagStats(
            pass_rate=round(len(tag_passing) / len(tag_results), 3),
            pass_count=len(tag_passing),
            total_count=len(tag_results),
        )

    return Report(
        run_id=run_id,
        test_set_path=test_set_path,
        total_cases=total,
        evaluated_cases=len(judged),
        skipped_cases=skipped,
        accuracy=_dim_stats("accuracy"),
        groundedness=_dim_stats("groundedness"),
        safety=_dim_stats("safety"),
        format=_dim_stats("format"),
        overall_pass_rate=round(len(overall_passing) / total, 3) if total else 0.0,
        overall_pass_count=len(overall_passing),
        by_tag=by_tag,
        timestamp=timestamp,
    )


# ── Terminal rendering ────────────────────────────────────────────────────────


def print_report(report: Report) -> None:
    """Render the report as a rich terminal table."""
    console.rule(
        f"[bold cyan]leeg-eval run — {report.timestamp[:19]}[/bold cyan]  "
        f"[dim]{report.test_set_path}[/dim]"
    )

    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("dimension", style="dim", min_width=14)
    table.add_column("score", justify="right")
    table.add_column("pass rate", justify="right")
    table.add_column("n_judged", justify="right")
    table.add_column("failing cases", style="dim")

    for dim_name in ("accuracy", "groundedness", "safety", "format"):
        dim: DimensionStats = getattr(report, dim_name)
        failing_str = ", ".join(dim.failing_cases) if dim.failing_cases else "—"
        pass_pct = f"{dim.pass_rate * 100:.0f}%"
        table.add_row(
            dim_name,
            f"{dim.mean_score:.2f}",
            pass_pct,
            str(dim.total_count),
            failing_str,
        )

    table.add_section()
    table.add_row(
        "[bold]OVERALL[/bold]",
        "—",
        f"[bold]{report.overall_pass_rate * 100:.0f}%[/bold]",
        f"{report.total_cases} total / {report.evaluated_cases} judged",
        "",
    )

    console.print(table)

    # Tag breakdown
    if report.by_tag:
        tag_parts = [
            f"[bold]{tag}[/bold]: {stats.pass_count}/{stats.total_count} "
            f"({stats.pass_rate * 100:.0f}%)"
            for tag, stats in sorted(report.by_tag.items())
        ]
        console.print("by tag:  " + " │ ".join(tag_parts))

    # One-line summary
    console.print(
        f"\n[green]PASS_RATE: {report.overall_pass_rate * 100:.0f}%[/green]"
        f" | accuracy: {report.accuracy.mean_score:.2f}"
        f" | groundedness: {report.groundedness.mean_score:.2f}"
        f" | safety: {report.safety.mean_score:.2f}"
        f" | format: {report.format.mean_score:.2f}"
        f" | ({report.overall_pass_count}/{report.total_cases} cases passed,"
        f" {report.skipped_cases} skipped deterministic)"
    )
    console.rule()


def print_comparison(report_a: Report, path_a: str, report_b: Report, path_b: str) -> None:
    """Render a side-by-side diff of two reports."""
    console.rule("[bold cyan]leeg-eval regression comparison[/bold cyan]")
    console.print(f"  A: [dim]{path_a}[/dim]  ({report_a.timestamp[:19]})")
    console.print(f"  B: [dim]{path_b}[/dim]  ({report_b.timestamp[:19]})\n")

    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("dimension", style="dim", min_width=14)
    table.add_column("score A", justify="right")
    table.add_column("score B", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("pass rate A", justify="right")
    table.add_column("pass rate B", justify="right")
    table.add_column("Δ pass rate", justify="right")

    for dim_name in ("accuracy", "groundedness", "safety", "format"):
        a: DimensionStats = getattr(report_a, dim_name)
        b: DimensionStats = getattr(report_b, dim_name)
        score_delta = b.mean_score - a.mean_score
        rate_delta = b.pass_rate - a.pass_rate

        def _fmt_delta(d: float) -> str:
            if d > 0.005:
                return f"[green]+{d:.3f}[/green]"
            if d < -0.005:
                return f"[red]{d:.3f}[/red]"
            return f"[dim]{d:+.3f}[/dim]"

        table.add_row(
            dim_name,
            f"{a.mean_score:.3f}",
            f"{b.mean_score:.3f}",
            _fmt_delta(score_delta),
            f"{a.pass_rate * 100:.0f}%",
            f"{b.pass_rate * 100:.0f}%",
            _fmt_delta(rate_delta),
        )

    table.add_section()
    overall_delta = report_b.overall_pass_rate - report_a.overall_pass_rate

    def _fmt_delta(d: float) -> str:
        if d > 0.005:
            return f"[green]+{d:.3f}[/green]"
        if d < -0.005:
            return f"[red]{d:.3f}[/red]"
        return f"[dim]{d:+.3f}[/dim]"

    table.add_row(
        "[bold]OVERALL[/bold]",
        "—", "—", "—",
        f"[bold]{report_a.overall_pass_rate * 100:.0f}%[/bold]",
        f"[bold]{report_b.overall_pass_rate * 100:.0f}%[/bold]",
        _fmt_delta(overall_delta),
    )

    console.print(table)
    console.rule()


# ── Persistence ───────────────────────────────────────────────────────────────


def save_report(
    report: Report,
    results: list[RunResult],
    output_dir: str = "reports/runs/",
) -> Path:
    """Save the report and all RunResults to a timestamped JSONL file.

    Line 1:   {"_type": "report_summary", ...}
    Lines 2+: {"_type": "run_result", ...}

    Returns the path to the saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.run_id}.jsonl"

    with out_path.open("w") as f:
        summary = {"_type": "report_summary", **report.model_dump()}
        f.write(json.dumps(summary) + "\n")
        for r in results:
            row = {"_type": "run_result", **r.model_dump()}
            f.write(json.dumps(row) + "\n")

    log.info("reporter.saved", path=str(out_path), total_cases=len(results))
    return out_path


def load_report(path: str) -> Report:
    """Load just the Report summary from the first line of a run JSONL file."""
    with open(path) as f:
        first_line = f.readline().strip()
    data = json.loads(first_line)
    data.pop("_type", None)
    return Report.model_validate(data)
