"""Build the final combined benchmark: easy + hard phish (tagged) + credential-benign.

One benchmark serves every RQ: RQ1/detection use the full set; RQ2 uses the
source=="phishpedia-hard" subset. Phish are brand-balanced and split into an easy
(unrelated-domain) and a hard (brand-in-domain lookalike) half; benign are login-like
credential pages so the comparison is legitimacy, not page type (see data audit).

Usage:
    .venv/bin/python scripts/build_combined.py --n-easy 365 --n-hard 300 \
        --n-benign 665 --calib-frac 0.25 --seed 21 --max-per-brand 12 --out data/phishsel_final
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


def brand_cap(recs: list[PageRecord], k: int, max_per_brand: int, rng: random.Random):
    by: dict[str, list[PageRecord]] = defaultdict(list)
    for r in recs:
        by[canonical_brand(r.brand) or "_"].append(r)
    pool: list[PageRecord] = []
    for items in by.values():
        rng.shuffle(items)
        pool.extend(items[:max_per_brand])
    rng.shuffle(pool)
    if len(pool) < k:
        chosen = {id(r) for r in pool}
        rest = [r for r in recs if id(r) not in chosen]
        rng.shuffle(rest)
        pool.extend(rest[: k - len(pool)])
    return pool[:k]


def stratified_split(records, calib_frac, rng):
    calib, test = [], []
    by_label: dict[Label, list[PageRecord]] = defaultdict(list)
    for r in records:
        by_label[r.label].append(r)
    for items in by_label.values():
        items = items[:]
        rng.shuffle(items)
        k = round(len(items) * calib_frac)
        calib += [r.model_copy(update={"split": "calibration"}) for r in items[:k]]
        test += [r.model_copy(update={"split": "test"}) for r in items[k:]]
    rng.shuffle(calib)
    rng.shuffle(test)
    return calib, test


def summarize(name, recs):
    lab = Counter(r.label.value for r in recs)
    src = Counter(r.source for r in recs if r.label is Label.PHISH)
    return f"{name}: n={len(recs)} {dict(lab)} | phish src {dict(src)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phish", type=Path, default=Path("data/phishsel/manifest_phish.jsonl"))
    ap.add_argument("--hard", type=Path, default=Path("data/phishsel/manifest_hard_phish.jsonl"))
    ap.add_argument("--benign", type=Path, default=Path("data/phishsel/manifest_benign_cred.jsonl"))
    ap.add_argument("--n-easy", type=int, default=365)
    ap.add_argument("--n-hard", type=int, default=300)
    ap.add_argument("--n-benign", type=int, default=665)
    ap.add_argument("--calib-frac", type=float, default=0.25)
    ap.add_argument("--max-per-brand", type=int, default=12)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", type=Path, default=Path("data/phishsel_final"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    hard = [r.model_copy(update={"source": "phishpedia-hard"}) for r in read_manifest(args.hard)]
    hard_ids = {r.page_id for r in hard}
    easy = [r for r in read_manifest(args.phish) if r.page_id not in hard_ids]  # exclude lookalikes
    benign = read_manifest(args.benign)

    easy_s = brand_cap(easy, args.n_easy, args.max_per_brand, rng)
    hard_s = brand_cap(hard, args.n_hard, args.max_per_brand, rng)
    benign_s = rng.sample(benign, args.n_benign)
    pool = easy_s + hard_s + benign_s

    calib, test = stratified_split(pool, args.calib_frac, rng)
    write_manifest(calib, args.out / "calibration.jsonl")
    write_manifest(test, args.out / "test.jsonl")

    lines = [f"Combined benchmark (seed={args.seed})",
             summarize("pool ", pool), summarize("calib", calib), summarize("test ", test)]
    n_hard_test = sum(1 for r in test if r.source == "phishpedia-hard")
    lines.append(f"RQ2 hard subset in test: {n_hard_test}")
    report = "\n".join(lines)
    (args.out / "stats.txt").write_text(report + "\n")
    print(report)
    print(f"\n[ok] wrote {args.out}/calibration.jsonl, test.jsonl, stats.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
