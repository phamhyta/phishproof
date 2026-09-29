"""RQ3 component ablation, re-scored from the cached panel (no agent re-run).

Re-runs the panel (cache hits) to recover A, G, and label-vote agreement per page, then
forms each ablation variant's score, fits a per-variant isotonic calibrator on the
calibration split, and evaluates AURC / FPR80 / ECE on test:
  - full                = A * G  (PhishProof)
  - - calibration       = same ranking, ECE on the RAW score (no kappa)
  - - groundedness      = A only (drop the tool-grounding factor)
  - - evidence-agreement= B4 * G (swap evidence agreement A for label-vote agreement)
The - diversity row comes from a separate same-family panel run (Pass 3).

Usage: .venv/bin/python scripts/rescore_ablation.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.panel import Panel
from phishproof.aggregate.gea import score_page
from phishproof.baselines.label_scores import b4_majority_vote
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import aurc, ece, fpr_at_coverage
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def score_split(panel, pages, ctx):
    rows = []
    for p in pages:
        outs = panel.run(p)  # cache hit
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        rows.append({
            "label": 1 if p.label.value == "phish" else 0,
            "verdict": 1 if r.verdict.value == "phish" else 0,
            "A": r.agreement, "G": r.groundedness, "gea": r.gea,
            "b4": b4_majority_vote(outs),
        })
    return rows


def variant_score(row, name):
    if name == "full" or name == "-calibration":
        return row["gea"]
    if name == "-groundedness":
        return row["A"]
    if name == "-evidence-agreement":
        return row["b4"] * row["G"]
    raise ValueError(name)


def evaluate(cal, te, name):
    cs = np.array([variant_score(r, name) for r in cal])
    ts = np.array([variant_score(r, name) for r in te])
    cc = np.array([r["label"] == r["verdict"] for r in cal])
    tc = np.array([r["label"] == r["verdict"] for r in te])
    yt = np.array([r["label"] for r in te])
    yp = np.array([r["verdict"] for r in te])
    a = aurc(ts, tc) * 100
    fpr = fpr_at_coverage(ts, yt, yp, 0.80) * 100
    if name == "-calibration":
        e = ece(ts, tc)  # raw score, no calibrator
    else:
        cal_obj = IsotonicCalibrator().fit(cs.tolist(), cc.tolist())
        e = ece(np.array(cal_obj.predict(ts.tolist())), tc)
    return a, fpr, e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    args = ap.parse_args()

    panel = Panel.from_config(load_panel())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())
    print("re-scoring calib + test from cache (ablation variants)...")
    cal = score_split(panel, read_manifest(args.data / "calibration.jsonl"), ctx)
    te = score_split(panel, read_manifest(args.data / "test.jsonl"), ctx)

    print(f"\n{'Variant':24} {'AURC':>7} {'FPR80':>7} {'ECE':>7}")
    for name in ["full", "-calibration", "-groundedness", "-evidence-agreement"]:
        a, fpr, e = evaluate(cal, te, name)
        print(f"{name:24} {a:7.2f} {fpr:7.2f} {e:7.3f}")
    print("(- diversity row: run Pass 3 same-family panel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
