"""Evaluate one method's per-page scores into the headline metric table (RQ1, tab_main).

Returns each metric as (point, lo, hi) from a paired bootstrap. Pass a shared boot_idx
across methods so the CIs are paired (same page draws) and differences are comparable.
"""

from __future__ import annotations

import numpy as np

from .bootstrap import ci_from_draws, paired_bootstrap_indices
from .metrics import (
    aurc,
    coverage_at_selective_accuracy,
    detection_metrics,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)

SELECTIVE_KEYS = ("AURC", "SelAcc80", "FPR80", "Cov99", "ECE")


def evaluate_method(
    scores,
    y_true,
    y_pred,
    trust=None,
    boot_idx: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    coverage: float = 0.80,
    target_acc: float = 0.99,
) -> dict[str, tuple[float, float, float]]:
    s = np.asarray(scores, dtype=float)
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    correct = yt == yp
    trust = s if trust is None else np.asarray(trust, dtype=float)
    if boot_idx is None:
        boot_idx = paired_bootstrap_indices(len(s), n_boot, seed)

    def metrics_on(idx):
        return {
            "AURC": aurc(s[idx], correct[idx]),
            "SelAcc80": selective_accuracy_at_coverage(s[idx], correct[idx], coverage),
            "FPR80": fpr_at_coverage(s[idx], yt[idx], yp[idx], coverage),
            "Cov99": coverage_at_selective_accuracy(s[idx], correct[idx], target_acc),
            "ECE": ece(trust[idx], correct[idx]),
        }

    point = metrics_on(np.arange(len(s)))
    draws = {k: np.empty(len(boot_idx)) for k in point}
    for b, idx in enumerate(boot_idx):
        m = metrics_on(idx)
        for k in m:
            draws[k][b] = m[k]

    return {k: (point[k], *ci_from_draws(draws[k])) for k in point}


def evaluate_detection(y_true, y_pred, boot_idx=None, n_boot=1000, seed=0):
    """Full-coverage detection metrics with CIs (tab_detect)."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if boot_idx is None:
        boot_idx = paired_bootstrap_indices(len(yt), n_boot, seed)
    point = detection_metrics(yt, yp)
    draws = {k: np.empty(len(boot_idx)) for k in point}
    for b, idx in enumerate(boot_idx):
        m = detection_metrics(yt[idx], yp[idx])
        for k in m:
            draws[k][b] = m[k]
    return {k: (point[k], *ci_from_draws(draws[k])) for k in point}
