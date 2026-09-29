"""Selective decision: pick the operating threshold, then act/abstain (C4, eq:objective).

A selective detector acts only when the trust score clears a threshold tau and abstains
otherwise. Thresholds are fit on the calibration split (never the test split):
  - target-risk:    smallest tau (max coverage) whose acted-set selective risk <= r  (Cov99)
  - target-coverage: the tau that acts on a target fraction of pages                 (SelAcc80)
"""

from __future__ import annotations

import numpy as np

from ..schema import GEAResult, Label, PageRecord


def correctness(results: list[GEAResult], labels: dict[str, Label]) -> list[bool]:
    """Per-result: did the panel's verdict match the page's true label?"""
    return [r.verdict == labels[r.page_id] for r in results]


def risk_coverage(scores, correct, threshold: float) -> tuple[float, float]:
    """(coverage, selective_risk) at a threshold. Risk = error rate among acted pages."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    acted = scores >= threshold
    cov = float(acted.mean()) if len(scores) else 0.0
    if acted.sum() == 0:
        return 0.0, 0.0
    risk = float(1.0 - correct[acted].mean())
    return cov, risk


def threshold_for_target_risk(scores, correct, target_risk: float) -> float:
    """Smallest threshold (=> max coverage) whose selective risk <= target_risk.

    If no threshold meets the target, return the one with the lowest risk (most
    conservative = the highest score).
    """
    scores = np.asarray(scores, dtype=float)
    candidates = sorted(set(scores.tolist()))
    best_thr, best_risk = candidates[-1] if candidates else 1.0, 1.0
    for thr in candidates:  # ascending => increasing coverage
        cov, risk = risk_coverage(scores, correct, thr)
        if cov == 0:
            continue
        if risk <= target_risk:
            return float(thr)  # first (smallest) thr that satisfies => max coverage
        if risk < best_risk:
            best_thr, best_risk = thr, risk
    return float(best_thr)


def threshold_for_coverage(scores, target_coverage: float) -> float:
    """Threshold that acts on ~target_coverage of pages (quantile of the score)."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return 1.0
    q = float(np.quantile(scores, 1.0 - target_coverage))
    return q


def apply_selection(
    results: list[GEAResult],
    threshold: float,
    calibrator=None,
) -> list[GEAResult]:
    """Set calibrated_trust (if a calibrator is given) and acted on each result."""
    out = []
    for r in results:
        trust = calibrator(r.gea) if calibrator is not None else None
        out.append(r.model_copy(update={
            "calibrated_trust": trust,
            "acted": bool(r.gea >= threshold),
        }))
    return out


def labels_from_pages(pages: list[PageRecord]) -> dict[str, Label]:
    return {p.page_id: p.label for p in pages}
