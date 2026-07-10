"""Run the PhishProof panel over a page manifest.

Wraps :class:`phishproof.pipeline.PhishProofRunner`. The panel implementation is
released with the paper; this entry point prints the release notice.
"""

from __future__ import annotations

import argparse
import sys

from phishproof._release import RELEASE_NOTICE


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a corpus with the PhishProof panel.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest of pages")
    parser.add_argument("--config", help="pipeline configuration (JSON)")
    parser.add_argument("--out", required=True, help="output result bundle (JSONL)")
    parser.parse_args()
    print(RELEASE_NOTICE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
