"""Ingest a Phishpedia sample folder tree into a PhishSel manifest (JSONL).

Phishpedia phish_sample_30k / benign_sample_30k both use the same layout:
    <raw_dir>/<folder>/{html.txt, shot.png, info.txt, coordinates.txt, ...}

Usage:
    .venv/bin/python scripts/ingest_phishpedia.py --raw data/raw --label phish \
        --out data/phishsel/manifest_phish.jsonl
    .venv/bin/python scripts/ingest_phishpedia.py --raw data/benign_raw --label benign \
        --out data/phishsel/manifest_benign.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import iter_phishpedia, write_manifest
from phishproof.schema import Label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="folder of Phishpedia sample dirs")
    ap.add_argument("--label", required=True, choices=["phish", "benign"])
    ap.add_argument("--out", required=True, type=Path, help="output manifest .jsonl")
    ap.add_argument("--limit", type=int, default=None, help="stop after N valid records (debug)")
    args = ap.parse_args()

    if not args.raw.is_dir():
        print(f"[FAIL] not a directory: {args.raw}")
        return 1

    label = Label(args.label)
    records = []
    limited = False
    for rec in iter_phishpedia(args.raw, label):
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            limited = True
            break

    write_manifest(records, args.out)
    print(f"[ok] {len(records)} {label.value} records -> {args.out}")
    if limited:
        print(f"     (stopped early at --limit {args.limit}; not a full scan)")
    else:
        total_dirs = sum(1 for p in args.raw.iterdir() if p.is_dir())
        print(f"     scanned {total_dirs} folders, skipped {total_dirs - len(records)} "
              f"(missing html/shot/info or bad url)")
    if records:
        with_brand = sum(1 for r in records if r.brand)
        print(f"     with brand annotation: {with_brand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
