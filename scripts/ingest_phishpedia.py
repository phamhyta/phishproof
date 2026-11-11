"""Build a manifest from a Phishpedia-style folder of captures."""

import argparse
from pathlib import Path

from phishproof.datasets import iter_phishpedia, write_manifest
from phishproof.schema import Label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=[label.value for label in Label])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = list(iter_phishpedia(args.raw, Label(args.label)))
    write_manifest(records, args.out)
    print(f"{len(records)} {args.label} pages -> {args.out}")


if __name__ == "__main__":
    main()
