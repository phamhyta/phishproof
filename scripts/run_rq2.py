"""RQ2 — right-label, wrong-reason: flip rate under an evidence-targeted perturbation.

Takes correctly-labeled PHISH pages, cloaks the cited form-action evidence (rewrites the
form to an on-brand-looking domain) while keeping the true label phish, re-runs the panel,
and measures how often the verdict FLIPS. The RQ2 claim: low-GEA pages (right label, weak/
disagreed evidence) flip far more than high-GEA pages — they are exactly the fragile
verdicts PhishProof abstains on, while confidence baselines act on them.

Usage:
    .venv/bin/python scripts/run_rq2.py --bundle results/bundle_pilot_v3.jsonl \
        --data data/phishsel_pilot --out results/rq2_pilot.json --limit 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.agents.panel import Panel
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.perturb import cloak_form_action
from phishproof.schema import Label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/rq2.json"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--hard-only", action="store_true")
    args = ap.parse_args()

    bundle = {json.loads(l)["page_id"]: json.loads(l)
              for l in args.bundle.read_text().splitlines() if l.strip()}
    pages = {p.page_id: p for p in read_manifest(args.data / "test.jsonl")}

    # Correctly-labeled phish pages, spread across the GEA range (sort, take evenly).
    # With --hard-only, restrict to the lookalike (brand-in-domain) subset where the RLWR
    # fragility gradient concentrates.
    correct_phish = [pid for pid, r in bundle.items()
                     if r["label"] == "phish" and r["verdict"] == "phish"
                     and pid in pages
                     and (not args.hard_only or r.get("is_hard"))]
    correct_phish.sort(key=lambda pid: bundle[pid]["gea"])
    if len(correct_phish) > args.limit:
        idx = np.linspace(0, len(correct_phish) - 1, args.limit).astype(int)
        correct_phish = [correct_phish[i] for i in idx]
    print(f"RQ2 on {len(correct_phish)} correctly-labeled phish pages")

    panel = Panel.from_config(load_panel())
    pert_dir = Path("results/perturbed")

    rows = []
    for i, pid in enumerate(correct_phish, 1):
        page = pages[pid]
        gea = bundle[pid]["gea"]
        perturbed = cloak_form_action(page, pert_dir)
        outs = panel.run(perturbed)
        new_verdict = Panel.majority_label(outs)
        flipped = new_verdict != Label.PHISH        # original verdict was phish
        rows.append({"page_id": pid, "gea": gea, "flipped": bool(flipped)})
        if i % 20 == 0:
            print(f"  {i}/{len(correct_phish)}  flips so far: {sum(r['flipped'] for r in rows)}")

    geas = np.array([r["gea"] for r in rows])
    flips = np.array([r["flipped"] for r in rows], dtype=float)

    # Partition by GEA tertiles: low (RLWR-like) vs high (well-grounded).
    lo_t, hi_t = np.quantile(geas, [1 / 3, 2 / 3])
    low = geas <= lo_t
    high = geas >= hi_t
    report = {
        "n": len(rows),
        "overall_flip_rate": float(flips.mean()),
        "low_gea": {"thr": float(lo_t), "n": int(low.sum()),
                    "flip_rate": float(flips[low].mean()) if low.any() else None},
        "high_gea": {"thr": float(hi_t), "n": int(high.sum()),
                     "flip_rate": float(flips[high].mean()) if high.any() else None},
    }
    # Per-decile trend (should fall monotonically as GEA rises).
    order = np.argsort(geas)
    deciles = np.array_split(order, 5)
    report["quintile_flip_rates"] = [
        {"gea_mean": float(geas[d].mean()), "flip_rate": float(flips[d].mean())}
        for d in deciles
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n=== RQ2 results ===")
    print(f"Overall flip rate: {report['overall_flip_rate']:.3f}")
    print(f"Low-GEA  (≤{lo_t:.2f}, n={report['low_gea']['n']}): "
          f"flip {report['low_gea']['flip_rate']:.3f}")
    print(f"High-GEA (≥{hi_t:.2f}, n={report['high_gea']['n']}): "
          f"flip {report['high_gea']['flip_rate']:.3f}")
    print("Quintiles (GEA↑ → flip↓ expected):")
    for q in report["quintile_flip_rates"]:
        print(f"   GEA~{q['gea_mean']:.2f}: flip {q['flip_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
