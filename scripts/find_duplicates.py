"""Assign near-duplicate groups across manifests and write them as a JSON map."""

import argparse
import json
from pathlib import Path

from phishproof.datasets import duplicate_groups, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hamming-radius", type=int, default=3)
    args = parser.parse_args()

    pages = {page.page_id: page for path in args.manifests for page in read_manifest(path)}
    groups = duplicate_groups(list(pages.values()), args.hamming_radius)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(groups))
    print(f"{len(groups)} pages in {len(set(groups.values()))} groups -> {args.out}")


if __name__ == "__main__":
    main()
