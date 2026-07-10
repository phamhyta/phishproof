"""Compute the selective-detection results (RQ1-RQ4) from a scored bundle.

Wraps :mod:`phishproof.eval`. The metric implementations are released with the
paper; this entry point prints the release notice.
"""

from __future__ import annotations

import argparse
import sys

from phishproof._release import RELEASE_NOTICE


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a scored PhishProof bundle.")
    parser.add_argument("--bundle", required=True, help="scored result bundle (JSONL)")
    parser.add_argument("--out", help="destination for the metrics report (JSON)")
    parser.parse_args()
    print(RELEASE_NOTICE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
