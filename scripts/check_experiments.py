"""Phase 5 end-to-end test: synthesize a results bundle, then run the RQ1 evaluator.

Generates a bundle where PhishProof's calibrated trust ranks correct pages well and the
baselines rank progressively worse, writes results/bundle_synthetic.jsonl, and invokes the
real run_experiments pipeline so the whole emit path is exercised without models/benign.

Run:  .venv/bin/python scripts/check_experiments.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def synth_bundle(n: int = 4000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    err = rng.random(n) < 0.08                      # shared panel error
    y_pred = np.where(err, 1 - y_true, y_true)
    correct = (y_pred == y_true).astype(float)

    def informative(strength):
        return np.clip(correct * rng.normal(0.5 + strength, 0.12, n)
                       + (1 - correct) * rng.normal(0.5 - strength, 0.12, n), 0, 1)

    gea = informative(0.30)        # PhishProof: strongest separation
    trust = np.clip(gea, 0, 1)     # pretend already calibrated
    b5 = informative(0.12)         # label-agreement: weak separation
    b4 = informative(0.08)
    b1 = informative(0.05)
    d3 = informative(0.10)

    bundle = []
    for i in range(n):
        bundle.append({
            "page_id": f"p{i}",
            "label": "phish" if y_true[i] else "benign",
            "verdict": "phish" if y_pred[i] else "benign",
            "gea": float(gea[i]),
            "calibrated_trust": float(trust[i]),
            "baselines": {"B1": float(b1[i]), "B4": float(b4[i]),
                          "B5": float(b5[i]), "D3": float(d3[i])},
        })
    return bundle


def main() -> int:
    out = ROOT / "results" / "bundle_synthetic.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle = synth_bundle()
    with out.open("w") as f:
        for r in bundle:
            f.write(json.dumps(r) + "\n")
    print(f"[ok] wrote synthetic bundle: {out} ({len(bundle)} pages)\n")

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_experiments.py"),
         "--bundle", str(out), "--n-boot", "300"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return 1

    data = json.loads((ROOT / "results" / "rq1_main.json").read_text())
    pp = data["methods"]["PhishProof"]["AURC"][0]
    best_b = data["best_baseline"]
    bb = data["methods"][best_b]["AURC"][0]
    ok = pp < bb
    print(f"[check] PhishProof AURC={pp*100:.2f} < best baseline {best_b} AURC={bb*100:.2f} ? {ok}")
    print("\nPhase 5 experiments OK" if ok else "\nPhase 5 — REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
