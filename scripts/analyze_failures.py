"""B/W1.7: failure-mode taxonomy of PhishProof's full-coverage errors.

Breaks the 81 errors on the 998-page test split into a small taxonomy with each mode's
count and mean GEA, so the paper can replace the one-paragraph "76 missed / 5 FP" summary
with a table. Cache-free: reads results/bundle_final.jsonl only.

Usage: .venv/bin/python scripts/analyze_failures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.eval.bundle import load_bundle

R = Path("results")


def has_brand(r) -> bool:
    return any(c["type"] == "brand_claim" for c in r.get("consensus_cues", []))


def n_consensus(r) -> int:
    return len(r.get("consensus_cues", []))


def main() -> int:
    bundle = load_bundle(R / "bundle_final.jsonl")
    fn = [r for r in bundle if r["label"] == "phish" and r["verdict"] == "benign"]
    fp = [r for r in bundle if r["label"] == "benign" and r["verdict"] == "phish"]

    fn_zero = [r for r in fn if n_consensus(r) == 0]
    fn_nobrand = [r for r in fn if n_consensus(r) > 0 and not has_brand(r)]
    fn_brand = [r for r in fn if has_brand(r)]

    def mg(rows):
        return float(np.mean([r["gea"] for r in rows])) if rows else 0.0

    def conf(rows):
        return sum(1 for r in rows if r["gea"] > 0.5)

    modes = [
        ("FN: no consensus cue", fn_zero),
        ("FN: cues but no brand", fn_nobrand),
        ("FN: brand in consensus", fn_brand),
        ("FP: false alarm", fp),
    ]
    rows = []
    print(f"{'mode':28}{'n':>5}{'meanGEA':>9}{'conf(>0.5)':>11}")
    for name, rs in modes:
        rows.append({"mode": name, "n": len(rs), "mean_gea": mg(rs), "confident": conf(rs)})
        print(f"{name:28}{len(rs):>5}{mg(rs):>9.3f}{conf(rs):>11}")
    total_err = len(fn) + len(fp)
    total_conf = sum(r["confident"] for r in rows)
    print(f"\ntotal errors={total_err}  FN={len(fn)}  FP={len(fp)}  confident errors={total_conf}")
    print(f"correct pages mean GEA={mg([r for r in bundle if r['verdict']==r['label']]):.3f}")

    (R / "failure_modes.json").write_text(json.dumps(
        {"modes": rows, "total_errors": total_err, "n_fn": len(fn), "n_fp": len(fp),
         "confident_errors": total_conf}, indent=2))
    print("\n[ok] wrote results/failure_modes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
