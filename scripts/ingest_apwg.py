"""Build a manifest from the APWG phishing4190 corpus."""

import argparse
from pathlib import Path

from phishproof.datasets import iter_apwg, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path, help="folder with phishing4190_2.csv")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = list(iter_apwg(args.raw))
    write_manifest(records, args.out)
    print(f"{len(records)} APWG pages -> {args.out}")


if __name__ == "__main__":
    main()
