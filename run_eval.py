#!/usr/bin/env python3
"""leeg-eval — CLI entrypoint.

Usage:
    # Run a full eval on a test set
    python run_eval.py --test-set test_sets/sms_flows.jsonl

    # Debug a single input interactively (prints full trace + judge reasoning)
    python run_eval.py --input "Bob is out Tuesday" \\
        --context '{"channel":"sms","team_id":1}' --debug

    # Regression comparison between two saved run files
    python run_eval.py --compare reports/runs/2026-03-01T12-00-00.jsonl \\
        reports/runs/2026-03-16T14-22-00.jsonl

    # Run a single experiment config and log to MLflow  (Phase 6)
    python run_eval.py --experiment experiments/configs/prompt_variants/few_shot.yaml

    # Run every experiment config sequentially (Phase 6)
    python run_eval.py --run-all-experiments

    # Launch the MLflow local UI (Phase 6)
    python run_eval.py --mlflow-ui
"""

import argparse
import asyncio
import json
import os
import sys

import httpx
import structlog
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger(__name__)


# ── Auth helpers ──────────────────────────────────────────────────────────────


async def _acquire_token(base_url: str) -> str:
    """Return a JWT, preferring LEEG_API_TOKEN; falling back to email/password login."""
    token = os.getenv("LEEG_API_TOKEN", "").strip()
    if token:
        return token

    email = os.getenv("LEEG_API_EMAIL", "").strip()
    password = os.getenv("LEEG_API_PASSWORD", "").strip()
    if not email or not password:
        print(
            "ERROR: Set LEEG_API_TOKEN, or both LEEG_API_EMAIL and LEEG_API_PASSWORD in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        try:
            resp = await client.post(
                "/api/auth/login",
                data={"username": email, "password": password},
            )
            resp.raise_for_status()
            token = resp.json()["access_token"]
            log.info("auth.token_acquired", email=email)
            return token
        except httpx.HTTPStatusError as exc:
            print(
                f"ERROR: Login failed ({exc.response.status_code}): {exc.response.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        except httpx.ConnectError:
            print(
                f"ERROR: Cannot connect to Leeg API at {base_url}. Is leeg-app running?",
                file=sys.stderr,
            )
            sys.exit(1)


def _make_client(base_url: str, token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )


def _make_judge():
    """Instantiate LLMJudge from environment variables, or return None."""
    from scorers.llm_judge import LLMJudge

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001").strip()
    if not api_key:
        log.warning("run_eval.no_judge", reason="ANTHROPIC_API_KEY not set — judge disabled")
        return None
    return LLMJudge(model=model, api_key=api_key)


# ── Mode implementations ──────────────────────────────────────────────────────


async def _run_test_set(test_set_path: str) -> None:
    """Full eval loop: load test set → run batch → report → save."""
    from reports.reporter import generate_report, print_report, save_report
    from runners.batch_runner import run_batch

    base_url = os.getenv("LEEG_API_URL", "http://localhost:8000")
    token = await _acquire_token(base_url)
    judge = _make_judge()

    async with _make_client(base_url, token) as client:
        try:
            results = await run_batch(test_set_path, client, judge=judge)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except httpx.ConnectError:
            print(
                f"ERROR: Cannot reach Leeg API at {base_url}. Is leeg-app running?",
                file=sys.stderr,
            )
            sys.exit(1)

    report = generate_report(results, test_set_path=test_set_path)
    print_report(report)
    saved_path = save_report(report, results)
    print(f"\nRun saved to: {saved_path}")


async def _run_single_input(raw_input: str, context_json: str, debug: bool) -> None:
    """Debug a single input through the pipeline."""
    from models import TestCase
    from runners.single_runner import run_single

    try:
        context = json.loads(context_json)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --context is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    test_case = TestCase(
        id="adhoc_001",
        input=raw_input,
        context=context,
        expected_intent="query",
        expected_action="none",
        expected_safe=True,
        tags=["adhoc"],
        scoring_rubric="Ad-hoc debug run — no formal rubric.",
    )

    base_url = os.getenv("LEEG_API_URL", "http://localhost:8000")
    token = await _acquire_token(base_url)
    judge = _make_judge() if debug else None

    async with _make_client(base_url, token) as client:
        try:
            await run_single(test_case, client, judge=judge, debug=debug)
        except httpx.ConnectError:
            print(
                f"ERROR: Cannot reach Leeg API at {base_url}. Is leeg-app running?",
                file=sys.stderr,
            )
            sys.exit(1)


def _run_compare(path_a: str, path_b: str) -> None:
    """Diff two saved run JSONL files and print a regression table."""
    from reports.reporter import load_report, print_comparison

    for path in (path_a, path_b):
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        report_a = load_report(path_a)
        report_b = load_report(path_b)
    except Exception as exc:
        print(f"ERROR: Could not load report: {exc}", file=sys.stderr)
        sys.exit(1)

    print_comparison(report_a, path_a, report_b, path_b)


async def _run_experiment(config_path: str) -> None:
    """Run a single experiment config and log to MLflow. (Phase 6)"""
    print(f"[leeg-eval] --experiment mode not yet implemented (Phase 6). Config: {config_path}")
    sys.exit(1)


async def _run_all_experiments() -> None:
    """Run all experiment configs sequentially. (Phase 6)"""
    print("[leeg-eval] --run-all-experiments not yet implemented (Phase 6).")
    sys.exit(1)


def _launch_mlflow_ui() -> None:
    """Launch the MLflow local UI. (Phase 6)"""
    import subprocess
    tracking_uri = "experiments/results"
    print(f"[leeg-eval] Launching MLflow UI (tracking store: {tracking_uri}) …")
    subprocess.run(["mlflow", "ui", "--backend-store-uri", tracking_uri], check=False)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="leeg-eval — evaluation and optimization framework for the Leeg AI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--test-set",
        metavar="PATH",
        help="Run the full eval loop against a JSONL test set and print the report.",
    )
    mode.add_argument(
        "--input",
        metavar="TEXT",
        help="Run a single input through the pipeline (use with --context and --debug).",
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Diff two saved run JSONL files and show per-dimension score changes.",
    )
    mode.add_argument(
        "--experiment",
        metavar="CONFIG",
        help="Run a single experiment config YAML and log results to MLflow.",
    )
    mode.add_argument(
        "--run-all-experiments",
        action="store_true",
        help="Run every config in experiments/configs/ sequentially and log all results.",
    )
    mode.add_argument(
        "--mlflow-ui",
        action="store_true",
        help="Launch the MLflow local UI for browsing experiment results.",
    )

    parser.add_argument(
        "--context",
        metavar="JSON",
        default='{"channel":"sms","team_id":1}',
        help='Pipeline context as a JSON string (used with --input).',
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the full pipeline trace and judge reasoning (used with --input).",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.test_set:
        asyncio.run(_run_test_set(args.test_set))

    elif args.input:
        asyncio.run(_run_single_input(args.input, args.context, args.debug))

    elif args.compare:
        _run_compare(args.compare[0], args.compare[1])

    elif args.experiment:
        asyncio.run(_run_experiment(args.experiment))

    elif args.run_all_experiments:
        asyncio.run(_run_all_experiments())

    elif args.mlflow_ui:
        _launch_mlflow_ui()

    return 0


if __name__ == "__main__":
    sys.exit(main())
