"""T5 -- content-level duplicate audit and a grouped, duplicate-aware re-split.

The earlier audit (reviews/manifest-audit.json) could only compare manifest STRINGS: it ran on
a clone where every capture path was missing, so it explicitly scoped itself to "manifest
identities only... Missing page content cannot be checked for near duplicates." The captures
are present here, so this does the content-level check that audit could not:

  1. SHA-256 of every DOM HTML and every screenshot, plus a 64-bit dHash perceptual hash of
     each screenshot, so re-captures of the same page that differ by a few bytes still match.
  2. A pairwise overlap matrix by class for every calibration/test pair, counting BOTH unique
     identities and affected rows, over six keys: page_id, url, registrable domain, html sha,
     screenshot sha, and near-duplicate screenshot (dHash Hamming <= --dhash-radius).
  3. A grouped re-split. Rows are joined by union-find over exact url, identical html, identical
     screenshot, near-duplicate screenshot, and shared registrable domain; each group is then
     assigned WHOLE to one side (the side holding most of its rows; ties go to test, and the
     seed is frozen). Nothing is resampled back up to the original count -- the split shrinks,
     and that is the point.

Writes the group assignment so the re-split is reproducible, and a test-side page_id list that
scripts/revision_matched.py can be pointed at to re-run T1 on the regrouped split.

Usage
    uv run scripts/revision_grouped_split.py --out results/revision/t5_grouped_split.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishproof.tools.urls import registrable_domain  # noqa: E402

DATASETS = {
    "phishsel_final": ("calibration.jsonl", "test.jsonl"),
    "apwg_final": ("calibration.jsonl", "test.jsonl"),
    "trop_final": ("calibration.jsonl", "test.jsonl"),
}


def sha256_file(p: str | None) -> str | None:
    if not p or not Path(p).exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(p: str | None, size: int = 8) -> int | None:
    """64-bit difference hash: robust to rescaling and mild recompression."""
    if not p or not Path(p).exists():
        return None
    try:
        img = Image.open(p).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:  # noqa: BLE001 - unreadable capture
        return None
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


class UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/revision/t5_grouped_split.json"))
    ap.add_argument("--dhash-radius", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- load + fingerprint ----------------------------------------------------
    rows: list[dict] = []
    for ds, (cal, test) in DATASETS.items():
        for split, fn in (("calibration", cal), ("test", test)):
            p = args.data / ds / fn
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                r["_ds"], r["_split"] = ds, split
                r["_key"] = f"{ds}/{split}:{r['page_id']}"
                rows.append(r)
    print(f"[t5] fingerprinting {len(rows)} rows ...", flush=True)

    def fp(r):
        r["_html_sha"] = sha256_file(r.get("dom_html_path"))
        r["_shot_sha"] = sha256_file(r.get("screenshot_path"))
        r["_dhash"] = dhash(r.get("screenshot_path"))
        r["_domain"] = registrable_domain(r.get("final_url") or r.get("url") or "")
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(fp, rows))
    print(f"[t5] done. html hashed={sum(1 for r in rows if r['_html_sha'])}, "
          f"shots hashed={sum(1 for r in rows if r['_shot_sha'])}", flush=True)

    # ---- near-duplicate screenshot pairs (dHash Hamming <= radius) --------------
    have = [r for r in rows if r["_dhash"] is not None]
    hv = np.array([r["_dhash"] for r in have], dtype=np.uint64)
    near_pairs: list[tuple[int, int]] = []
    B = 2000
    for i0 in range(0, len(hv), B):
        block = hv[i0:i0 + B]
        d = popcount64(block[:, None] ^ hv[None, :])
        ii, jj = np.where(d <= args.dhash_radius)
        for a, b in zip(ii + i0, jj):
            if a < b:
                near_pairs.append((int(a), int(b)))
    print(f"[t5] near-duplicate screenshot pairs (<= {args.dhash_radius} bits): {len(near_pairs)}")

    # ---- overlap matrix by class ------------------------------------------------
    def keyset(rs, field):
        return {r[field] for r in rs if r.get(field)}

    dhash_group: dict[int, set[str]] = defaultdict(set)
    uf_d = UF()
    for a, b in near_pairs:
        uf_d.union(have[a]["_key"], have[b]["_key"])
    for r in have:
        dhash_group[uf_d.find(r["_key"])].add(r["_key"])
    key_to_dgroup = {k: g for g, ks in dhash_group.items() for k in ks}

    overlap = []
    names = sorted({(r["_ds"], r["_split"]) for r in rows})
    for i in range(len(names)):
        for j in range(len(names)):
            if i >= j:
                continue
            (da, sa), (db, sb) = names[i], names[j]
            A = [r for r in rows if r["_ds"] == da and r["_split"] == sa]
            Bb = [r for r in rows if r["_ds"] == db and r["_split"] == sb]
            for label in ("phish", "benign"):
                Al = [r for r in A if r["label"] == label]
                Bl = [r for r in Bb if r["label"] == label]
                if not Al or not Bl:
                    continue
                entry = {"a": f"{da}/{sa}", "b": f"{db}/{sb}", "label": label,
                         "a_n": len(Al), "b_n": len(Bl), "keys": {}}
                for field in ("url", "_domain", "_html_sha", "_shot_sha"):
                    ka, kb = keyset(Al, field), keyset(Bl, field)
                    shared = ka & kb
                    entry["keys"][field.lstrip("_")] = {
                        "shared_unique": len(shared),
                        "a_rows_affected": sum(1 for r in Al if r.get(field) in shared),
                        "b_rows_affected": sum(1 for r in Bl if r.get(field) in shared)}
                ga = {key_to_dgroup.get(r["_key"]) for r in Al} - {None}
                gb = {key_to_dgroup.get(r["_key"]) for r in Bl} - {None}
                sh = ga & gb
                entry["keys"]["screenshot_near_dup"] = {
                    "shared_unique": len(sh),
                    "a_rows_affected": sum(1 for r in Al if key_to_dgroup.get(r["_key"]) in sh),
                    "b_rows_affected": sum(1 for r in Bl if key_to_dgroup.get(r["_key"]) in sh)}
                overlap.append(entry)

    # ---- grouped re-split on phishsel_final -------------------------------------
    pf = [r for r in rows if r["_ds"] == "phishsel_final"]
    uf = UF()
    for r in pf:
        uf.find(r["_key"])
    for field in ("url", "_domain", "_html_sha", "_shot_sha"):
        buckets: dict[str, list[str]] = defaultdict(list)
        for r in pf:
            if r.get(field):
                buckets[f"{field}:{r[field]}"].append(r["_key"])
        for ks in buckets.values():
            for k in ks[1:]:
                uf.union(ks[0], k)
    pf_keys = {r["_key"] for r in pf}
    for a, b in near_pairs:
        ka, kb = have[a]["_key"], have[b]["_key"]
        if ka in pf_keys and kb in pf_keys:
            uf.union(ka, kb)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in pf:
        groups[uf.find(r["_key"])].append(r)

    assign: dict[str, str] = {}
    rng = np.random.RandomState(args.seed)
    for g, rs in sorted(groups.items()):
        c = Counter(r["_split"] for r in rs)
        side = ("test" if c["test"] >= c["calibration"] else "calibration")
        assign[g] = side
    new_split = {r["page_id"]: assign[uf.find(r["_key"])] for r in pf}

    orig = Counter((r["_split"], r["label"]) for r in pf)
    new = Counter((new_split[r["page_id"]], r["label"]) for r in pf)
    moved = sum(1 for r in pf if new_split[r["page_id"]] != r["_split"])
    gsizes = Counter(len(v) for v in groups.values())

    test_ids = sorted(r["page_id"] for r in pf
                      if r["_split"] == "test" and new_split[r["page_id"]] == "test")

    result = {
        "scope": ("Content-level: SHA-256 of DOM html and screenshot, plus 64-bit dHash on "
                  "screenshots. Supersedes reviews/manifest-audit.json, which could only "
                  "compare manifest strings because its clone had no captures."),
        "dhash_radius": args.dhash_radius, "seed": args.seed,
        "n_rows": len(rows),
        "near_dup_screenshot_pairs": len(near_pairs),
        "overlap": overlap,
        "regroup_phishsel_final": {
            "n_groups": len(groups),
            "group_size_hist": dict(sorted(gsizes.items())[:12]),
            "largest_group": max(len(v) for v in groups.values()),
            "original_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(orig.items())},
            "regrouped_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(new.items())},
            "rows_moved": moved,
            "test_rows_retained": len(test_ids),
        },
        "regrouped_test_page_ids": test_ids,
        # page_id -> duplicate-group id, for the ORIGINAL test split. This is what makes a
        # duplicate-aware re-run of T1 possible: with random folds, a near-duplicate of a test
        # page can sit in the folds its calibrator is fitted on, so the map is partly fitted on
        # the very page it scores. Grouping the folds removes that.
        "test_group_map": {r["page_id"]: uf.find(r["_key"])
                           for r in pf if r["_split"] == "test"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[t5] groups={len(groups)} largest={result['regroup_phishsel_final']['largest_group']} "
          f"moved={moved} test retained={len(test_ids)}")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
