"""Phase 5 smoke test: selective metrics + bootstrap CIs on synthetic methods.

Builds two methods that share the SAME panel verdicts (same full-coverage error) but
rank pages differently — a GEA-like score that ranks correct pages above errors, and a
label-concordance baseline that ranks ~randomly w.r.t. correctness. The selective metrics
must reward the better ranking (lower AURC, higher Cov99/SelAcc80) — the RQ1 story.

Run:  .venv/bin/python scripts/check_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.eval import (
    detection_metrics,
    evaluate_method,
    paired_bootstrap_indices,
    significant,
)
from phishproof.eval.report import print_table


def main() -> int:
    rng = np.random.default_rng(0)
    n = 4000
    y_true = rng.integers(0, 2, n)                 # balanced phish/benign
    # panel makes ~8% errors (shared by both methods -> same detection metrics)
    err = rng.random(n) < 0.08
    y_pred = np.where(err, 1 - y_true, y_true)
    correct = (y_pred == y_true).astype(float)

    # GEA-like: ranks correct pages above errors (informative trust)
    gea = np.clip(correct * rng.normal(0.8, 0.12, n) + (1 - correct) * rng.normal(0.3, 0.12, n), 0, 1)
    # baseline: trust ~ independent of correctness (label concordance, uninformative here)
    base = rng.uniform(0, 1, n)

    boot = paired_bootstrap_indices(n, n_boot=500, seed=1)
    m_gea = evaluate_method(gea, y_true, y_pred, boot_idx=boot)
    m_base = evaluate_method(base, y_true, y_pred, boot_idx=boot)

    keys = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")
    print(print_table({"PhishProof(GEA)": m_gea, "baseline(rand)": m_base}, keys))

    det = detection_metrics(y_true, y_pred)
    print(f"\n[detection] full-coverage: acc={det['accuracy']:.3f} P={det['precision']:.3f} "
          f"R={det['recall']:.3f} F1={det['f1']:.3f}")

    # checks: better ranking => lower AURC, higher Cov99, and the AURC gap is significant
    aurc_gea, aurc_base = m_gea["AURC"][0], m_base["AURC"][0]
    cov_gea, cov_base = m_gea["Cov99"][0], m_base["Cov99"][0]
    sig = significant(m_gea["AURC"][1:], m_base["AURC"][1:])
    print(f"\n[check] AURC  GEA={aurc_gea:.3f} < baseline={aurc_base:.3f} ? {aurc_gea < aurc_base}")
    print(f"[check] Cov99 GEA={cov_gea:.3f} > baseline={cov_base:.3f} ? {cov_gea > cov_base}")
    print(f"[check] AURC gap significant (non-overlapping CIs)? {sig}")

    ok = aurc_gea < aurc_base and cov_gea > cov_base and sig
    print("\nPhase 5 metrics OK" if ok else "\nPhase 5 — REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
