"""Selective + detection metrics, bootstrap CIs, reporting (Phase 5)."""

from .bootstrap import bootstrap_ci, paired_bootstrap_indices, significant
from .evaluate import SELECTIVE_KEYS, evaluate_detection, evaluate_method
from .metrics import (
    aurc,
    coverage_at_selective_accuracy,
    detection_metrics,
    ece,
    fpr_at_coverage,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)

__all__ = [
    "aurc",
    "risk_coverage_curve",
    "selective_accuracy_at_coverage",
    "fpr_at_coverage",
    "coverage_at_selective_accuracy",
    "ece",
    "detection_metrics",
    "evaluate_method",
    "evaluate_detection",
    "SELECTIVE_KEYS",
    "bootstrap_ci",
    "paired_bootstrap_indices",
    "significant",
]
