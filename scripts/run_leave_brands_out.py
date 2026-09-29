"""W2.2: leave-brands-out generalization (a $0, reproducible cross-distribution test).

Live-crawling fresh PhishTank pages is not reproducible in-session (phishing URLs die within
hours), so instead of a new corpus we test the harder generalization question directly: does
the calibrated selective layer transfer to brands it has NEVER seen during calibration?

We pool all 1,330 pages (reusing the cached GEA scores in results/hfgea_features.npz), split
the brands into two disjoint halves, fit the isotonic calibrator + Cov99 threshold on pages
whose brand is in the SEEN half, and evaluate on pages whose brand is in the UNSEEN half. The
benign pages (no brand) are split at random. Averaged over several brand partitions, this
reports how the selective metrics hold up on unseen brands vs the in-distribution split.

Usage: .venv/bin/python scripts/run_leave_brands_out.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.calibration import IsotonicCalibrator
from phishproof.data_io import read_manifest
from phishproof.eval.metrics import (
    aurc,
    coverage_at_selective_accuracy,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)
from phishproof.tools.brands import canonical_brand


def load_pool(data: Path):
    """Method-D canonical: pool the 4020 test pages from results/bundle_D.jsonl (gea = method-D
    mean per-type agreement); brand from the test manifest. Parameter-free gea is brand-agnostic,
    so this tests whether the calibrated selective layer transfers to unseen brands."""
    rows = [json.loads(l) for l in Path("results/bundle_D.jsonl").read_text().splitlines() if l.strip()]
    pages = {p.page_id: p for p in read_manifest(data / "test.jsonl")}
    gea = np.array([r["gea"] for r in rows])
    label = np.array([1 if r["label"] == "phish" else 0 for r in rows])
    verdict = np.array([1 if r["verdict"] == "phish" else 0 for r in rows])
    correct = (label == verdict)
    brand = []
    for r in rows:
        p = pages.get(r["page_id"])
        brand.append(canonical_brand(p.brand) if (p and p.brand) else None)
    return gea, correct, label, verdict, brand


def evaluate(gea, correct, label, verdict, seen_mask):
    cal = IsotonicCalibrator().fit(gea[seen_mask].tolist(), correct[seen_mask].tolist())
    un = ~seen_mask
    trust = np.array(cal.predict(gea[un].tolist()))
    c, yt, yp = correct[un], label[un], verdict[un]
    return {
        "n": int(un.sum()),
        "acc": float(c.mean()),
        "AURC": aurc(trust, c) * 100,
        "SelAcc80": selective_accuracy_at_coverage(trust, c, 0.80) * 100,
        "FPR80": fpr_at_coverage(trust, yt, yp, 0.80) * 100,
        "Cov99": coverage_at_selective_accuracy(trust, c, 0.99),
        "ECE": ece(trust, c),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    gea, correct, label, verdict, brand = load_pool(args.data)
    brands = sorted({b for b in brand if b})
    print(f"pool={len(gea)} pages, {len(brands)} unique phish brands")

    runs = []
    for seed in range(args.folds):
        rng = np.random.default_rng(seed)
        perm = list(brands)
        rng.shuffle(perm)
        seen_brands = set(perm[: len(perm) // 2])
        # phish page seen iff its brand is in the seen half; benign split at random
        seen_mask = np.zeros(len(gea), bool)
        for i, b in enumerate(brand):
            if b is not None:
                seen_mask[i] = b in seen_brands
            else:
                seen_mask[i] = rng.random() < 0.5
        runs.append(evaluate(gea, correct, label, verdict, seen_mask))

    keys = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")
    print(f"\nLeave-brands-out generalization ({args.folds} brand partitions):")
    print(f"{'metric':10}{'mean':>9}{'std':>8}")
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"{k:10}{vals.mean():9.2f}{vals.std():8.2f}")
    print(f"\nmean unseen-brand test size = {np.mean([r['n'] for r in runs]):.0f} pages, "
          f"acc = {np.mean([r['acc'] for r in runs]):.3f}")
    print("(compare to in-distribution tab_main: AURC 2.3, SelAcc80 97.0, ECE 0.015, Cov99 0.25)")

    Path("results/leave_brands_out.json").write_text(json.dumps(
        {"folds": runs, "summary": summary}, indent=2))
    print("\n[ok] wrote results/leave_brands_out.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
