"""Ingest APWG phishing4190 (PhishingEval, Zenodo 14668190) into a PhishProof manifest.

Layout (different from Phishpedia):
    phishing4190/
      phishing4190_2.csv          # brand,domain,scr_path,html_path
      apwg_sample_4190/
        <YYYY-MM>/<YYYY-MM-DD>/<id>.html
                                  <id>.png

`scr_path` and `html_path` in the csv are relative to `phishing4190/`. Brand is the
gold target column (e.g. 'Shopify', 'PayPal'). Each phish gets a synthetic URL
http://<domain>/ since the corpus does not retain the original phishing URL.

Usage:
    .venv/bin/python scripts/ingest_apwg.py --raw data/_dl/extracted/phishing4190 \
        --out data/_manifests/apwg_phish.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import slug, write_manifest
from phishproof.schema import Label, PageRecord


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path,
                    help="folder containing phishing4190_2.csv and apwg_sample_4190/")
    ap.add_argument("--out", required=True, type=Path, help="output manifest .jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    csv_path = args.raw / "phishing4190_2.csv"
    if not csv_path.exists():
        print(f"[FAIL] not found: {csv_path}")
        return 1

    records: list[PageRecord] = []
    skipped = 0
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            brand = (row.get("brand") or "").strip() or None
            domain = (row.get("domain") or "").strip()
            scr_rel = (row.get("scr_path") or "").strip()
            html_rel = (row.get("html_path") or "").strip()
            if not (scr_rel and html_rel):
                skipped += 1; continue
            shot = (args.raw / scr_rel).resolve()
            html = (args.raw / html_rel).resolve()
            if not (shot.exists() and html.exists()):
                skipped += 1; continue
            # page id from html stem (e.g. 2021-07-15-20049 -> stable across runs)
            page_id = slug(Path(html_rel).stem)
            records.append(PageRecord(
                page_id=page_id,
                url=f"http://{domain}/" if domain else f"http://apwg/{page_id}",
                label=Label.PHISH,
                dom_html_path=str(html),
                screenshot_path=str(shot),
                raw_dir=str(shot.parent),
                brand=brand,
                source="apwg",
            ))
            if args.limit and len(records) >= args.limit:
                break

    write_manifest(records, args.out)
    with_brand = sum(1 for r in records if r.brand)
    print(f"[ok] {len(records)} apwg phish records -> {args.out}")
    print(f"     with brand: {with_brand}; skipped (missing file): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
