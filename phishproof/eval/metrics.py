"""Selective + detection metrics (experiments.tex §Tasks and metrics).

Every method emits a per-page trust score; we evaluate that score as a selective
detector. All selective functions take aligned arrays:
    scores  : trust score per page (higher = more trusted)
    y_true  : true label (1 = phish, 0 = benign)
    y_pred  : predicted label (the panel's majority verdict)
"correct" = (y_pred == y_true).

Metrics: AURC, SelAcc@cov, FPR@cov, Cov@selective-accuracy, ECE, and full-coverage
detection accuracy / precision / recall / F1.
"""

from __future__ import annotations

import numpy as np


def _as_arrays(scores, correct):
    return np.asarray(scores, dtype=float), np.asarray(correct, dtype=bool)


def risk_coverage_curve(scores, correct):
    """Return (coverage, risk) arrays by acting on the top-k highest-score pages.

    coverage[k] = k/n ; risk[k] = error rate among those top-k pages.
    """
    s, c = _as_arrays(scores, correct)
    n = len(s)
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-s, kind="stable")          # most-trusted first
    err = (~c[order]).astype(float)
    cum_err = np.cumsum(err)
    k = np.arange(1, n + 1)
    return k / n, cum_err / k


def aurc(scores, correct) -> float:
    """Area under the risk-coverage curve (empirical AURC). Lower is better."""
    _cov, risk = risk_coverage_curve(scores, correct)
    return float(risk.mean()) if len(risk) else 0.0


def _threshold_at_coverage(scores, coverage: float) -> float:
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return 1.0
    return float(np.quantile(s, 1.0 - coverage))


def _acted_mask(scores, coverage: float):
    s = np.asarray(scores, dtype=float)
    thr = _threshold_at_coverage(s, coverage)
    mask = s >= thr
    # trim ties so coverage ~ target (np.quantile can include extra equal-score pages)
    if mask.sum() > round(coverage * len(s)) and (s == thr).sum() > 1:
        keep = round(coverage * len(s))
        idx = np.argsort(-s, kind="stable")[:keep]
        m = np.zeros(len(s), dtype=bool)
        m[idx] = True
        return m
    return mask


def selective_accuracy_at_coverage(scores, correct, coverage: float) -> float:
    _s, c = _as_arrays(scores, correct)
    mask = _acted_mask(scores, coverage)
    return float(c[mask].mean()) if mask.sum() else 0.0


def fpr_at_coverage(scores, y_true, y_pred, coverage: float) -> float:
    """False-positive rate among acted pages: FP / (FP + TN), phish = positive."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    mask = _acted_mask(scores, coverage)
    yt_a, yp_a = yt[mask], yp[mask]
    neg = ~yt_a
    if neg.sum() == 0:
        return 0.0
    fp = (yp_a & neg).sum()
    return float(fp / neg.sum())


def coverage_at_selective_accuracy(scores, correct, target_acc: float = 0.99) -> float:
    """Max coverage whose selective accuracy >= target_acc (the Cov99 operating point)."""
    cov, risk = risk_coverage_curve(scores, correct)
    acc = 1.0 - risk
    ok = acc >= target_acc
    return float(cov[ok].max()) if ok.any() else 0.0


def ece(trust, correct, n_bins: int = 10) -> float:
    """Expected calibration error of a trust score against correctness."""
    t = np.asarray(trust, dtype=float)
    c = np.asarray(correct, dtype=float)
    if len(t) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi) if hi < 1.0 else (t >= lo) & (t <= hi)
        if m.sum():
            e += (m.mean()) * abs(c[m].mean() - t[m].mean())
    return float(e)


def detection_metrics(y_true, y_pred) -> dict[str, float]:
    """Full-coverage binary detection: accuracy, precision, recall, F1 (phish = positive)."""
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    tp = int((yp & yt).sum())
    fp = int((yp & ~yt).sum())
    fn = int((~yp & yt).sum())
    tn = int((~yp & ~yt).sum())
    acc = (tp + tn) / len(yt) if len(yt) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
