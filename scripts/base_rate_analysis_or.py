"""Deployment realism: precision (PPV) vs base rate from the 70B bundle (cache-only, no LLM).

Reviewers note the balanced 50/50 test inflates selective numbers: at real phishing prevalence a
small FPR floods operators. We reweight the measured within-class rates (TPR, FPR are
prevalence-independent) to realistic prevalences and report PPV, for two operating modes:
  - full coverage: auto-act on every phish verdict;
  - selective: auto-act only when calibrated trust >= tau (else escalate to human review).

At the selective operating point 0 / 2005 benign pages are auto-blocked, so we report the
conservative precision using the Wilson 95% upper bound on that zero-count FPR (the review asked
for Wilson intervals on zero/small cells).

Usage:
    .venv/bin/python scripts/base_rate_analysis_or.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.calibration import IsotonicCalibrator

TAU = 0.9651


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre + halfwidth


def main() -> int:
    cal = IsotonicCalibrator.from_dict(json.loads(Path("results/calibrator_or.json").read_text()))
    rows = [json.loads(l) for l in open("results/bundle_or.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("verdict") is not None]
    y = np.array([1 if r["label"] == "phish" else 0 for r in rows])
    yh = np.array([1 if r["verdict"] == "phish" else 0 for r in rows])
    trust = []
    for r in rows:
        ag = r.get("agents", [])
        vlm = next((x for x in ag if x.get("id") == "agent_c_vision"), None)
        c = vlm["confidence"] if (vlm and vlm.get("confidence") is not None) else 0.5
        vv = vlm["verdict"] if vlm else r["verdict"]
        txt = [x for x in ag if x.get("id") != "agent_c_vision"]
        a = sum(1 for x in txt if x.get("verdict") == vv) / len(txt) if txt else 0.0
        trust.append(cal(c + 1e-3 * a))
    trust = np.array(trust)
    P, Nn = (y == 1), (y == 0)
    out = {"n": len(rows), "tau": TAU, "modes": {}}

    def prec(pi, tpr, fpr):
        d = pi * tpr + (1 - pi) * fpr
        return pi * tpr / d if d > 0 else float("nan")

    for label, mask in [("full_coverage", np.ones(len(y), bool)), ("selective", trust >= TAU)]:
        block = (yh == 1) & mask
        fp, nneg = int((block & Nn).sum()), int(Nn.sum())
        tp, npos = int((block & P).sum()), int(P.sum())
        tpr, fpr = tp / npos, fp / nneg
        fpr_up = wilson_upper(fp, nneg)
        rec = {"coverage": float(mask.mean()), "tpr_block": tpr, "fpr_block": fpr,
               "fp": fp, "n_benign": nneg, "fpr_wilson_upper": fpr_up, "ppv": {}}
        for pi in (0.05, 0.01, 0.001):
            rec["ppv"][str(pi)] = {"point": 100 * prec(pi, tpr, fpr),
                                   "conservative": 100 * prec(pi, tpr, fpr_up)}
        out["modes"][label] = rec

    Path("results/_or/base_rate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nwrote results/_or/base_rate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
