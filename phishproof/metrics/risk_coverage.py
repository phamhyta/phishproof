"""Selective-prediction and detection metrics.

Verbalized confidences are coarse, so many pages tie. Every ranking metric here is the
expectation over the orders of tied pages, which does not depend on the input order.
"""

from __future__ import annotations

import numpy as np


def _expected_cumsum(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Expected running sum of ``values`` when pages are taken by score, with ties in random order."""
    order = np.argsort(-scores, kind="stable")
    ranked_scores, ranked_values = scores[order], values[order]
    result = np.empty(len(scores), dtype=float)
    start, running = 0, 0.0
    while start < len(scores):
        end = start
        while end < len(scores) and ranked_scores[end] == ranked_scores[start]:
            end += 1
        mean_value = ranked_values[start:end].mean()
        result[start:end] = running + mean_value * np.arange(1, end - start + 1)
        running += ranked_values[start:end].sum()
        start = end
    return result


def risk_coverage_curve(scores, correct) -> tuple[np.ndarray, np.ndarray]:
    """Coverage k/n and expected risk among the k highest-ranked pages, for every k."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(scores) == 0:
        return np.array([]), np.array([])
    taken = np.arange(1, len(scores) + 1)
    return taken / len(scores), _expected_cumsum(scores, (~correct).astype(float)) / taken


def aurc(scores, correct) -> float:
    _, risk = risk_coverage_curve(scores, correct)
    return float(risk.mean()) if len(risk) else 0.0


def selective_accuracy_at_coverage(scores, correct, coverage: float) -> float:
    _, risk = risk_coverage_curve(scores, correct)
    if not len(risk):
        return 0.0
    return float(1.0 - risk[max(round(coverage * len(risk)), 1) - 1])


def fpr_at_coverage(scores, y_true, y_pred, coverage: float) -> float:
    """Expected false positives over expected benign pages among the acted pages."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if len(scores) == 0:
        return 0.0
    index = max(round(coverage * len(scores)), 1) - 1
    negatives = _expected_cumsum(scores, (~y_true).astype(float))[index]
    false_positives = _expected_cumsum(scores, (y_pred & ~y_true).astype(float))[index]
    return float(false_positives / negatives) if negatives else 0.0


def coverage_at_selective_accuracy(scores, correct, target_accuracy: float = 0.99) -> float:
    coverage, risk = risk_coverage_curve(scores, correct)
    meets_target = 1.0 - risk >= target_accuracy
    return float(coverage[meets_target].max()) if meets_target.any() else 0.0


def ece(confidence, correct, n_bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence >= low) & (confidence < high if high < 1.0 else confidence <= high)
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def detection_metrics(y_true, y_pred) -> dict[str, float]:
    """Full-coverage accuracy, precision, recall and F1, with phishing as the positive class."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
