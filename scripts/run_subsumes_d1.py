"""Measure D1-as-tool subsumes: swap HtmlBrandDetector for PhishpediaDetector and
re-score GEA from cached panel outputs (no agent re-run).

Fits a per-variant isotonic calibrator on the calibration split, evaluates AURC on test.

Usage:
    .venv/bin/python scripts/run_subsumes_d1.py
    .venv/bin/python scripts/run_subsumes_d1.py --cache results/phishpedia_d1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.panel import Panel
from phishproof.aggregate.gea import score_page
from phishproof.calibration import IsotonicCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import (
    aurc,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.detectors.phishpedia import PhishpediaDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def score_split(panel, pages, ctx):
    rows = []
    for p in pages:
        outs = panel.run(p)
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        rows.append({
            "label": 1 if p.label.value == "phish" else 0,
            "verdict": 1 if r.verdict.value == "phish" else 0,
            "gea": r.gea,
        })
    return rows


def evaluate_split(cal_rows, test_rows):
    cal_scores = np.array([r["gea"] for r in cal_rows])
    cal_correct = np.array([r["label"] == r["verdict"] for r in cal_rows])
    test_scores = np.array([r["gea"] for r in test_rows])
    test_correct = np.array([r["label"] == r["verdict"] for r in test_rows])
    yt = np.array([r["label"] for r in test_rows])
    yp = np.array([r["verdict"] for r in test_rows])
    calibrator = IsotonicCalibrator().fit(cal_scores.tolist(), cal_correct.tolist())
    trust = np.array(calibrator.predict(test_scores.tolist()))
    return {
        "AURC": aurc(test_scores, test_correct) * 100,
        "SelAcc80": selective_accuracy_at_coverage(test_scores, test_correct, 0.80) * 100,
        "FPR80": fpr_at_coverage(test_scores, yt, yp, 0.80) * 100,
        "ECE": ece(trust, test_correct),
        "n_test": len(test_rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--cache", type=Path, default=Path("results/phishpedia_d1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/subsumes_d1.json"))
    args = ap.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"missing {args.cache} — run import_phishpedia_d1.py first")

    panel = Panel.from_config(load_panel())
    logo = CLIPLogoEmbedder()
    ctx_html = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=logo)
    ctx_d1 = GroundingContext(detector=PhishpediaDetector(args.cache), logo_embedder=logo)

    print("Re-scoring calibration + test (HtmlBrand vs Phishpedia brand grounder)...")
    cal_pages = read_manifest(args.data / "calibration.jsonl")
    test_pages = read_manifest(args.data / "test.jsonl")

    cal_html = score_split(panel, cal_pages, ctx_html)
    test_html = score_split(panel, test_pages, ctx_html)
    cal_d1 = score_split(panel, cal_pages, ctx_d1)
    test_d1 = score_split(panel, test_pages, ctx_d1)

    html_m = evaluate_split(cal_html, test_html)
    d1_m = evaluate_split(cal_d1, test_d1)

    out = {
        "n_test": len(test_pages),
        "html_brand": html_m,
        "phishpedia_d1": d1_m,
        "delta_AURC": d1_m["AURC"] - html_m["AURC"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nSubsumes ablation (n={len(test_pages)} test):")
    print(f"  HtmlBrandDetector   AURC {html_m['AURC']:.2f}")
    print(f"  PhishpediaDetector  AURC {d1_m['AURC']:.2f}")
    print(f"  delta AURC {out['delta_AURC']:+.2f}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
