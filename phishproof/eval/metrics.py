"""Selective-prediction metrics.

The definitions are standard and public; the implementations are released with
the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..types import Label


def aurc(scores: Sequence[float], correct: Sequence[int]) -> float:
    """Area under the risk-coverage curve (x100); lower is better."""
    pending("eval.metrics.aurc")


def selective_accuracy(scores: Sequence[float], correct: Sequence[int], coverage: float) -> float:
    """Accuracy over the most-confident ``coverage`` fraction (for example 0.80)."""
    pending("eval.metrics.selective_accuracy")


def fpr_at_coverage(
    scores: Sequence[float],
    correct: Sequence[int],
    labels: Sequence[Label],
    coverage: float,
) -> float:
    """False-positive rate within the acted-on fraction."""
    pending("eval.metrics.fpr_at_coverage")


def coverage_at_risk(scores: Sequence[float], correct: Sequence[int], max_risk: float) -> float:
    """Largest coverage whose selective risk stays under ``max_risk`` (e.g. 0.01)."""
    pending("eval.metrics.coverage_at_risk")


def expected_calibration_error(
    probs: Sequence[float],
    correct: Sequence[int],
    bins: int = 15,
) -> float:
    """Expected calibration error over equal-width probability bins."""
    pending("eval.metrics.expected_calibration_error")
