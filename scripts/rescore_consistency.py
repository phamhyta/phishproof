"""Re-score a bundle with the brand-domain consistency cue (no agent re-run).

Re-runs the panel (cached -> instant) to recover agent outputs, re-scores GEA with
add_consistency=True, re-fits the isotonic calibrator on the calibration split, and writes
a new bundle that keeps the existing baselines (incl. faithful B6) but updates gea +
calibrated_trust. Then run run_experiments.py on the new bundle.

Usage:
    .venv/bin/python scripts/rescore_consistency.py --data data/phishsel_hard_big \
        --bundle results/bundle_hard_big_logo.jsonl --out results/bundle_hard_big_consist.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.panel import Panel
from phishproof.aggregate.gea import score_page
from phishproof.calibration import IsotonicCalibrator, threshold_for_target_risk
from phishproof.config import load_experiment, load_panel
from phishproof.data_io import read_manifest
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.registry import GroundingContext


def score_split(panel, pages, ctx):
    return {p.page_id: score_page(panel.run(p), p, ctx, add_consistency=True) for p in pages}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    exp = load_experiment()
    panel = Panel.from_config(load_panel())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=None)

    calib = read_manifest(args.data / "calibration.jsonl")
    test = read_manifest(args.data / "test.jsonl")
    test_labels = {p.page_id: p.label for p in test}

    print(f"re-scoring {len(calib)} calib + {len(test)} test (consistency cue)...")
    cal = score_split(panel, calib, ctx)
    cal_lab = {p.page_id: p.label for p in calib}
    cal_scores = [r.gea for r in cal.values()]
    cal_correct = [r.verdict == cal_lab[pid] for pid, r in cal.items()]
    calibrator = IsotonicCalibrator().fit(cal_scores, cal_correct)
    tau = threshold_for_target_risk(cal_scores, cal_correct, exp.target_selective_risk)

    te = score_split(panel, test, ctx)

    # keep existing baselines (incl faithful B6); update gea + calibrated_trust
    existing = {json.loads(l)["page_id"]: json.loads(l)
                for l in args.bundle.read_text().splitlines() if l.strip()}
    rows = []
    for pid, r in te.items():
        base = existing.get(pid, {})
        rows.append({
            "page_id": pid,
            "label": test_labels[pid].value,
            "verdict": r.verdict.value,
            "gea": r.gea,
            "calibrated_trust": calibrator(r.gea),
            "baselines": base.get("baselines", {}),
        })

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[ok] wrote {args.out} (tau={tau:.3f})")
    print(f"     next: .venv/bin/python scripts/run_experiments.py --bundle {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
