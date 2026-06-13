from __future__ import annotations

import argparse
import json
import sys

from .io import read_pages
from .pipeline import PhishProofRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishproof")
    parser.add_argument("--input", required=True, help="JSONL file with page records")
    parser.add_argument("--config", help="JSON config for the public scaffold")
    parser.add_argument("--output", help="JSONL destination for predictions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = PhishProofRunner.from_config(args.config)

    if args.output:
        count = runner.run_file(args.input, args.output)
        print(f"wrote {count} predictions to {args.output}", file=sys.stderr)
        return 0

    for page in read_pages(args.input):
        print(json.dumps(runner.predict(page).to_json(), sort_keys=True))
    return 0
