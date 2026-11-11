"""Build a manifest from the KnowPhish TR-OP corpus (openphish_5000)."""

import argparse
from pathlib import Path

from phishproof.datasets import iter_trop, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = list(iter_trop(args.raw))
    write_manifest(records, args.out)
    print(f"{len(records)} TR-OP pages -> {args.out}")


if __name__ == "__main__":
    main()
