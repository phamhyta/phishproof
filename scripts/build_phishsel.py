"""Build the PhishSel benchmark: balanced subsample + stratified calib/test split.

Combines a phish manifest and a benign manifest, draws a class-balanced subsample of
--n-pages (deterministic given --seed), then splits into a small labeled calibration
split and a held-out test split (stratified by label). All thresholds + the isotonic
calibrator are fit on calibration; every metric is reported on test.

Usage:
    .venv/bin/python scripts/build_phishsel.py \
        --phish data/phishsel/manifest_phish.jsonl \
        --benign data/phishsel/manifest_benign.jsonl \
        --n-pages 4000 --calib-frac 0.15 --seed 0 --out data/phishsel
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import read_manifest, write_manifest
from phishproof.schema import Label, PageRecord
from phishproof.tools.brands import canonical_brand


def _brand_capped(phish: list[PageRecord], per_class: int, max_per_brand: int | None,
                  rng: random.Random) -> list[PageRecord]:
    """Sample per_class phish, capping pages per brand so no brand dominates."""
    if not max_per_brand:
        return rng.sample(phish, per_class)
    by_brand: dict[str, list[PageRecord]] = defaultdict(list)
    for r in phish:
        by_brand[canonical_brand(r.brand) or "unknown"].append(r)
    pool: list[PageRecord] = []
    for items in by_brand.values():
        rng.shuffle(items)
        pool.extend(items[:max_per_brand])
    rng.shuffle(pool)
    if len(pool) < per_class:
        # cap too tight: backfill from the leftover (still brand-diverse-ish)
        chosen = set(id(r) for r in pool)
        leftover = [r for r in phish if id(r) not in chosen]
        rng.shuffle(leftover)
        pool.extend(leftover[: per_class - len(pool)])
    return pool[:per_class]


def balanced_sample(
    phish: list[PageRecord], benign: list[PageRecord], n_pages: int, rng: random.Random,
    max_per_brand: int | None = None,
) -> list[PageRecord]:
    per_class = n_pages // 2
    if len(phish) < per_class or len(benign) < per_class:
        raise SystemExit(
            f"[FAIL] need {per_class}/class but have phish={len(phish)} benign={len(benign)}"
        )
    return _brand_capped(phish, per_class, max_per_brand, rng) + rng.sample(benign, per_class)


def stratified_split(
    records: list[PageRecord], calib_frac: float, rng: random.Random
) -> tuple[list[PageRecord], list[PageRecord]]:
    calib, test = [], []
    by_label: dict[Label, list[PageRecord]] = {Label.PHISH: [], Label.BENIGN: []}
    for r in records:
        by_label[r.label].append(r)
    for label, items in by_label.items():
        items = items[:]
        rng.shuffle(items)
        k = round(len(items) * calib_frac)
        for r in items[:k]:
            calib.append(r.model_copy(update={"split": "calibration"}))
        for r in items[k:]:
            test.append(r.model_copy(update={"split": "test"}))
    return calib, test


def summarize(name: str, records: list[PageRecord]) -> str:
    labels = Counter(r.label.value for r in records)
    brands = Counter(r.brand for r in records if r.brand)
    top = ", ".join(f"{b}:{c}" for b, c in brands.most_common(5))
    return f"{name}: n={len(records)} {dict(labels)} | top brands: {top}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phish", required=True, type=Path)
    ap.add_argument("--benign", required=True, type=Path)
    ap.add_argument("--n-pages", type=int, default=4000)
    ap.add_argument("--calib-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-brand", type=int, default=None,
                    help="cap phish pages per brand for diversity (e.g. 15)")
    ap.add_argument("--out", type=Path, default=Path("data/phishsel"))
    args = ap.parse_args()

    for p in (args.phish, args.benign):
        if not p.exists():
            print(f"[FAIL] manifest not found: {p}")
            if "benign" in p.name:
                print("       -> You still need benign pages. See note in build output.")
            return 1

    phish = read_manifest(args.phish)
    benign = read_manifest(args.benign)
    rng = random.Random(args.seed)

    sample = balanced_sample(phish, benign, args.n_pages, rng, args.max_per_brand)
    calib, test = stratified_split(sample, args.calib_frac, rng)
    # Shuffle so any prefix (--limit) is a balanced phish/benign mix, not class-ordered.
    rng.shuffle(calib)
    rng.shuffle(test)

    write_manifest(calib, args.out / "calibration.jsonl")
    write_manifest(test, args.out / "test.jsonl")

    lines = [
        f"PhishSel built (seed={args.seed}, n_pages={args.n_pages})",
        summarize("pool   ", sample),
        summarize("calib  ", calib),
        summarize("test   ", test),
    ]
    report = "\n".join(lines)
    (args.out / "stats.txt").write_text(report + "\n")
    print(report)
    print(f"\n[ok] wrote {args.out}/calibration.jsonl, test.jsonl, stats.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
