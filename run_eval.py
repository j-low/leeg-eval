#!/usr/bin/env python3
"""leeg-eval — CLI entrypoint.

Usage:
    python run_eval.py --test-set test_sets/sms_flows.jsonl
    python run_eval.py --input "Bob is out Tuesday" --context '{"channel":"sms"}' --debug
    python run_eval.py --compare reports/runs/2026-03-01.jsonl reports/runs/2026-03-16.jsonl
    python run_eval.py --experiment experiments/configs/prompt_variants/few_shot.yaml
    python run_eval.py --run-all-experiments
    python run_eval.py --mlflow-ui
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval.py",
        description="leeg-eval — evaluation and optimization framework for the Leeg AI pipeline",
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
        default="{}",
        help='Pipeline context as a JSON string (used with --input). Default: "{}"',
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
        print(f"[leeg-eval] --test-set mode not yet implemented (Phase 5). Target: {args.test_set}")
        return 1

    if args.input:
        print(f"[leeg-eval] --input mode not yet implemented (Phase 5). Input: {args.input!r}")
        return 1

    if args.compare:
        print(f"[leeg-eval] --compare mode not yet implemented (Phase 5). Files: {args.compare}")
        return 1

    if args.experiment:
        print(f"[leeg-eval] --experiment mode not yet implemented (Phase 6). Config: {args.experiment}")
        return 1

    if args.run_all_experiments:
        print("[leeg-eval] --run-all-experiments not yet implemented (Phase 6).")
        return 1

    if args.mlflow_ui:
        print("[leeg-eval] --mlflow-ui not yet implemented (Phase 6).")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
