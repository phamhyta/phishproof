"""Convert a Phishpedia-style corpus into a PhishProof page manifest.

Reads the ``{info.txt, shot.png, html.txt}`` per-page layout and emits a JSONL
manifest of page records. The ingestion body is released with the paper; this
entry point prints the release notice.
"""

from __future__ import annotations

import argparse
import sys

from phishproof._release import RELEASE_NOTICE


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a Phishpedia-style corpus.")
    parser.add_argument("--src", required=True, help="corpus root directory")
    parser.add_argument("--out", required=True, help="output manifest (JSONL)")
    parser.parse_args()
    print(RELEASE_NOTICE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
