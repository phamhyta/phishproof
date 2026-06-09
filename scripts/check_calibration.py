"""Phase 4 smoke test: isotonic calibration, selective thresholds, baselines.

Uses synthetic (score, correctness) data — no model or benign data needed. Verifies:
  - the calibrator's reported trust tracks empirical accuracy (ECE drops vs raw score)
  - a target-risk threshold actually holds the selective risk on held-out data
  - B1/B2/B4/B5/B6 baselines produce sensible per-page scores

Run:  .venv/bin/python scripts/check_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.baselines import compute_panel_baselines
from phishproof.calibration import (
    IsotonicCalibrator,
    risk_coverage,
    threshold_for_target_risk,
)
from phishproof.schema import AgentOutput, Label


def ece(scores, correct, n_bins=10) -> float:
    scores = np.asarray(scores)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (scores >= lo) & (scores < hi if hi < 1 else scores <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - scores[m].mean())
    return float(e)


def main() -> int:
    rng = np.random.default_rng(0)
    n = 6000
    q = rng.uniform(0, 1, n)                       # latent page quality
    correct = rng.binomial(1, q).astype(bool)      # correctness ~ Bernoulli(q)
    # Displayed GEA score is a monotonic *distortion* of q (like an over/under-confident
    # model), so the raw score is miscalibrated and isotonic has something to fix.
    score = np.clip(q ** 1.7 + rng.normal(0, 0.05, n), 0, 1)

    # split calibration / test
    k = int(0.25 * n)
    cal_s, cal_c = score[:k], correct[:k]
    te_s, te_c = score[k:], correct[k:]

    # 1) calibration: ECE of raw score vs calibrated trust on test
    cal = IsotonicCalibrator().fit(cal_s.tolist(), cal_c.tolist())
    te_trust = np.asarray(cal.predict(te_s.tolist()))
    ece_raw, ece_cal = ece(te_s, te_c), ece(te_trust, te_c)
    print(f"[calibration] ECE raw={ece_raw:.3f} -> calibrated={ece_cal:.3f} "
          f"({'better' if ece_cal < ece_raw else 'WORSE'})")

    # 2) target-risk threshold fit on calibration, checked on test (does it transfer?)
    target = 0.10
    thr = threshold_for_target_risk(cal_s.tolist(), cal_c.tolist(), target)
    cov_cal, risk_cal = risk_coverage(cal_s, cal_c, thr)
    cov_te, risk_te = risk_coverage(te_s, te_c, thr)
    # "smallest threshold meeting target" is optimistically biased, so allow finite-sample slack.
    transfers = risk_te <= target + 0.05 and cov_te > 0.05
    print(f"[selective]  target_risk={target}  tau={thr:.3f}")
    print(f"             calib: coverage={cov_cal:.2f} risk={risk_cal:.3f}")
    print(f"             test : coverage={cov_te:.2f} risk={risk_te:.3f} "
          f"(operating point {'transfers' if transfers else 'DRIFTS'})")

    # 3) calibrator dict round-trip
    cal2 = IsotonicCalibrator.from_dict(cal.to_dict())
    assert abs(cal2(0.5) - cal(0.5)) < 1e-6
    print(f"[roundtrip]  kappa(0.5)={cal(0.5):.3f} restored={cal2(0.5):.3f}")

    # 4) baselines on synthetic panel outputs
    def mk(pid, labels, confs):
        return [AgentOutput(agent_id=f"a{i}", verdict=lb, confidence=cf)
                for i, (lb, cf) in enumerate(zip(labels, confs))]

    P, B = Label.PHISH, Label.BENIGN
    outputs_by_page = {
        "unanimous": mk("unanimous", [P, P, P], [0.95, 0.92, 0.9]),
        "split2to1": mk("split2to1", [P, P, B], [0.7, 0.6, 0.8]),
        "tie_ish":   mk("tie_ish",   [P, B, B], [0.55, 0.6, 0.5]),
    }
    bl = compute_panel_baselines(outputs_by_page)
    print("[baselines]  per-page trust scores:")
    for pid in outputs_by_page:
        cells = "  ".join(f"{b}={bl[b][pid]:.2f}" for b in ("B1", "B2", "B4", "B5", "B6"))
        print(f"             {pid:10} {cells}")

    ok = ece_cal <= ece_raw + 1e-6 and transfers
    print("\nPhase 4 OK" if ok else "\nPhase 4 — REVIEW (a check did not hold)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
