"""Metric summaries with bootstrap intervals, each reported as (point, low, high)."""

from __future__ import annotations

import numpy as np

from .bootstrap import ci_from_draws, paired_bootstrap_indices
from .risk_coverage import (
    aurc,
    coverage_at_selective_accuracy,
    detection_metrics,
    ece,
    fpr_at_coverage,
    selective_accuracy_at_coverage,
)


def _with_intervals(metrics_on, n: int, draws) -> dict[str, tuple[float, float, float]]:
    point = metrics_on(np.arange(n))
    samples = {name: np.empty(len(draws)) for name in point}
    for position, indices in enumerate(draws):
        for name, value in metrics_on(indices).items():
            samples[name][position] = value
    return {name: (point[name], *ci_from_draws(samples[name])) for name in point}


def summarize_selective(
    scores,
    y_true,
    y_pred,
    confidence=None,
    draws=None,
    coverage: float = 0.80,
    target_accuracy: float = 0.99,
):
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    correct = y_true == y_pred
    confidence = scores if confidence is None else np.asarray(confidence, dtype=float)
    draws = paired_bootstrap_indices(len(scores)) if draws is None else draws

    def metrics_on(indices):
        return {
            "AURC": aurc(scores[indices], correct[indices]),
            "SelAcc": selective_accuracy_at_coverage(scores[indices], correct[indices], coverage),
            "FPR": fpr_at_coverage(scores[indices], y_true[indices], y_pred[indices], coverage),
            "Cov": coverage_at_selective_accuracy(
                scores[indices], correct[indices], target_accuracy
            ),
            "ECE": ece(confidence[indices], correct[indices]),
        }

    return _with_intervals(metrics_on, len(scores), draws)


def summarize_detection(y_true, y_pred, draws=None):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    draws = paired_bootstrap_indices(len(y_true)) if draws is None else draws
    return _with_intervals(
        lambda indices: detection_metrics(y_true[indices], y_pred[indices]), len(y_true), draws
    )
