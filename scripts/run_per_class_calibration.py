"""A1: per-class calibration vs global isotonic + score-asymmetry diagnostic.

Re-scores the calibration split from cache (no API), fits BOTH the current global
isotonic calibrator and the new per-class calibrator (separate curve for predicted-phish
vs predicted-benign), then evaluates each on the held-out test bundle. Also dumps the
score-asymmetry numbers that explain the Cov99 loss to B6.

Outputs:
  results/per_class_calibration.json   (metrics + asymmetry)
  results/calibrator_per_class.json    (the fitted per-class calibrator)

Usage: .venv/bin/python scripts/run_per_class_calibration.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.agents.panel import Panel
from phishproof.calibration import IsotonicCalibrator, PerClassCalibrator
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.eval.bundle import load_bundle
from phishproof.eval.metrics import (
    aurc,
    coverage_at_selective_accuracy,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext


def score_calib(data: Path):
    """Re-derive (gea, verdict, label) for the calibration split from cache."""
    panel = Panel.from_config(load_panel())
    ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=CLIPLogoEmbedder())
    pages = read_manifest(data / "calibration.jsonl")
    gea, verdict, label = [], [], []
    for i, p in enumerate(pages, 1):
        outs = panel.run(p)
        r = score_page(outs, p, ctx, add_consistency=True, relax_perceptual=True)
        gea.append(r.gea)
        verdict.append(r.verdict.value)
        label.append(p.label.value)
        if i % 100 == 0:
            print(f"  calib {i}/{len(pages)}", flush=True)
    return np.array(gea), verdict, label


def metrics_row(name, trust, correct, y_true, y_pred):
    return {
        "method": name,
        "AURC": aurc(trust, correct) * 100,
        "SelAcc80": selective_accuracy_at_coverage(trust, correct, 0.80) * 100,
        "FPR80": fpr_at_coverage(trust, y_true, y_pred, 0.80) * 100,
        "Cov99": coverage_at_selective_accuracy(trust, correct, 0.99),
        "ECE": ece(trust, correct),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_final.jsonl"))
    args = ap.parse_args()

    print("re-scoring calibration split from cache...")
    cal_gea, cal_verdict, cal_label = score_calib(args.data)
    cal_correct = [v == l for v, l in zip(cal_verdict, cal_label)]

    # fit both calibrators on the SAME calibration split
    iso = IsotonicCalibrator().fit(cal_gea.tolist(), cal_correct)
    pcc = PerClassCalibrator().fit(cal_gea.tolist(), cal_correct, cal_verdict)

    # test bundle
    bundle = load_bundle(args.bundle)
    gea = np.array([r["gea"] for r in bundle])
    verdict = [r["verdict"] for r in bundle]
    label = [r["label"] for r in bundle]
    correct = np.array([v == l for v, l in zip(verdict, label)])
    y_true = np.array([1 if l == "phish" else 0 for l in label])
    y_pred = np.array([1 if v == "phish" else 0 for v in verdict])

    trust_iso = np.array(iso.predict(gea.tolist()))
    trust_pcc = np.array(pcc.predict(gea.tolist(), verdict))

    rows = [
        metrics_row("global isotonic (current)", trust_iso, correct, y_true, y_pred),
        metrics_row("per-class isotonic (A1)", trust_pcc, correct, y_true, y_pred),
        metrics_row("raw GEA (no calib)", gea, correct, y_true, y_pred),
    ]

    # score-asymmetry: fraction of correct pages by class reaching trust>0.8
    def frac_high(trust, mask, thr=0.8):
        return float((trust[mask] > thr).mean()) if mask.any() else 0.0

    phish_correct = (y_true == 1) & correct
    benign_correct = (y_true == 0) & correct
    asym = {
        "global": {
            "phish_correct_mean": float(trust_iso[phish_correct].mean()),
            "phish_correct_frac>0.8": frac_high(trust_iso, phish_correct),
            "benign_correct_mean": float(trust_iso[benign_correct].mean()),
            "benign_correct_frac>0.8": frac_high(trust_iso, benign_correct),
        },
        "per_class": {
            "phish_correct_mean": float(trust_pcc[phish_correct].mean()),
            "phish_correct_frac>0.8": frac_high(trust_pcc, phish_correct),
            "benign_correct_mean": float(trust_pcc[benign_correct].mean()),
            "benign_correct_frac>0.8": frac_high(trust_pcc, benign_correct),
        },
    }

    print(f"\n{'method':28}{'AURC':>8}{'SelAcc80':>10}{'FPR80':>8}{'Cov99':>8}{'ECE':>8}")
    for r in rows:
        print(f"{r['method']:28}{r['AURC']:8.2f}{r['SelAcc80']:10.2f}"
              f"{r['FPR80']:8.2f}{r['Cov99']:8.3f}{r['ECE']:8.3f}")
    print("\nscore asymmetry (fraction of CORRECT pages reaching calibrated trust>0.8):")
    for k, d in asym.items():
        print(f"  {k:10} phish:{d['phish_correct_frac>0.8']*100:5.1f}%  "
              f"benign:{d['benign_correct_frac>0.8']*100:5.1f}%  "
              f"(mean phish {d['phish_correct_mean']:.3f} / benign {d['benign_correct_mean']:.3f})")

    out = {"metrics": rows, "asymmetry": asym}
    Path("results/per_class_calibration.json").write_text(json.dumps(out, indent=2))
    Path("results/calibrator_per_class.json").write_text(json.dumps(pcc.to_dict()))
    print("\n[ok] wrote results/per_class_calibration.json + calibrator_per_class.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
