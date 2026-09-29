"""Ingest KnowPhish TR-OP into a PhishProof manifest.

Layout (Phishpedia-compatible per page, brand encoded in folder name):
    openphish_5000/<brand>+<host>+<YYYY_MM_DD>+<n>/
        html.txt    shot.png    info.txt (= URL)    input_url.txt
    tranco_5000/+<host>+<YYYY_MM_DD>+<n>/  (benign — typically homepages)

Differs from Phishpedia: info.txt holds the bare URL (no brand); brand for phish
must be parsed from the folder-name prefix before '+'.

Phish only (label=phish). The native Tranco benigns are homepages — we DROP them
and pair login benigns from the Phishpedia credential pool (per plan).

Usage:
    .venv/bin/python scripts/ingest_trop.py --raw data/_dl/extracted/openphish_5000 \
        --out data/_manifests/trop_phish.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import parse_phishpedia_info, slug, write_manifest
from phishproof.schema import Label, PageRecord


def brand_from_folder(name: str) -> str | None:
    """openphish_5000 folder pattern: '<brand>+<host>+<YYYY_MM_DD>+<n>'.

    Returns brand if the prefix before the first '+' is non-empty; else None
    (matches the empty-prefix benign-style folders).
    """
    head, _, _ = name.partition("+")
    head = head.strip()
    # filter empties and placeholders like '?'
    return head if (head and head not in {"?", "-", "_", "unknown"}) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path,
                    help="folder of TR-OP page subdirs (e.g. data/_dl/extracted/openphish_5000)")
    ap.add_argument("--out", required=True, type=Path, help="output manifest .jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.raw.is_dir():
        print(f"[FAIL] not a directory: {args.raw}")
        return 1

    records: list[PageRecord] = []
    skipped = 0
    for folder in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        html = folder / "html.txt"
        shot = folder / "shot.png"
        if not (html.exists() and shot.exists()):
            skipped += 1; continue
        info = folder / "info.txt"
        meta = parse_phishpedia_info(info) if info.exists() else {}
        url = meta.get("url") or f"http://{folder.name}"
        records.append(PageRecord(
            page_id=slug(folder.name),
            url=url,
            label=Label.PHISH,
            dom_html_path=str(html),
            screenshot_path=str(shot),
            raw_dir=str(folder),
            brand=brand_from_folder(folder.name),  # TR-OP-specific: brand in folder name
            source="trop",
        ))
        if args.limit and len(records) >= args.limit:
            break

    write_manifest(records, args.out)
    with_brand = sum(1 for r in records if r.brand)
    print(f"[ok] {len(records)} trop phish records -> {args.out}")
    print(f"     with brand: {with_brand}; skipped (missing html/shot): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
