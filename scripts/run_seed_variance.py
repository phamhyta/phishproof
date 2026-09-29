"""5-seed re-split robustness ($0): the paper claims "re-partitioning the pages five ways
(stratified 85/15) leaves all metrics stable" -- this quantifies it.

For each of 5 seeds we re-partition the 4020 method-D test pages (results/bundle_D.jsonl) into a
fresh stratified 15% calibration / 85% evaluation split, refit the isotonic calibrator on the
calibration part, and recompute the selective metrics on the evaluation part. AURC uses the raw
parameter-free gea (headline convention); SelAcc/FPR/Cov99/ECE use the refit calibrated trust.

Usage: .venv/bin/python scripts/run_seed_variance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.calibration import IsotonicCalibrator
from phishproof.eval.metrics import (
    aurc,
    coverage_at_selective_accuracy,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)

KEYS = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")


def main() -> int:
    rows = [json.loads(l) for l in Path("results/bundle_D.jsonl").read_text().splitlines() if l.strip()]
    gea = np.array([r["gea"] for r in rows])
    label = np.array([1 if r["label"] == "phish" else 0 for r in rows])
    verdict = np.array([1 if r["verdict"] == "phish" else 0 for r in rows])
    correct = label == verdict
    n = len(rows)

    runs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        calib = np.zeros(n, bool)
        for cls in (0, 1):                              # stratified 15% calib / 85% eval
            idx = np.where(label == cls)[0]
            rng.shuffle(idx)
            calib[idx[: int(0.15 * len(idx))]] = True
        te = ~calib
        cal = IsotonicCalibrator().fit(gea[calib].tolist(), correct[calib].tolist())
        trust = np.array(cal.predict(gea[te].tolist()))
        runs.append({
            "AURC": aurc(gea[te], correct[te]) * 100,   # raw gea (headline convention)
            "SelAcc80": selective_accuracy_at_coverage(trust, correct[te], 0.80) * 100,
            "FPR80": fpr_at_coverage(trust, label[te], verdict[te], 0.80) * 100,
            "Cov99": coverage_at_selective_accuracy(trust, correct[te], 0.99),
            "ECE": ece(trust, correct[te]),
        })

    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs]))} for k in KEYS}
    print("5-seed re-split (stratified 15/85), method-D bundle_D:")
    for k in KEYS:
        print(f"  {k:9} {summary[k]['mean']:8.3f} ± {summary[k]['std']:.3f}")
    Path("results/_D/seed_variance.json").write_text(json.dumps({"folds": runs, "summary": summary}, indent=2))
    print("\n[ok] wrote results/_D/seed_variance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
