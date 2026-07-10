"""Command-line interface for PhishProof.

The argument surface is public so the workflow is legible; running any command
prints the release notice because the implementation is withheld until the paper
is accepted.
"""

from __future__ import annotations

import argparse
import sys

from ._release import RELEASE_NOTICE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishproof",
        description="Trustworthy, explainable, selective phishing detection.",
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="score a JSONL file of pages")
    run.add_argument("--input", required=True, help="JSONL file with page records")
    run.add_argument("--config", help="pipeline configuration (JSON)")
    run.add_argument("--output", help="JSONL destination for predictions")

    ev = sub.add_parser("evaluate", help="compute selective-detection metrics")
    ev.add_argument("--bundle", required=True, help="scored result bundle (JSONL)")

    cal = sub.add_parser("calibrate", help="fit the trust calibrator on a split")
    cal.add_argument("--bundle", required=True, help="scored calibration split (JSONL)")
    cal.add_argument("--output", required=True, help="calibrator destination (JSON)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help(sys.stderr)
        return 2
    print(RELEASE_NOTICE, file=sys.stderr)
    print(
        f"(the '{args.command}' command runs the implementation released upon acceptance)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
