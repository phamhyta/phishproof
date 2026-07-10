"""Evaluation: selective-prediction metrics and the adversarial study."""

from .bootstrap import bootstrap_ci, paired_test
from .evaluate import compare, evaluate_bundle
from .metrics import (
    aurc,
    coverage_at_risk,
    expected_calibration_error,
    fpr_at_coverage,
    selective_accuracy,
)
from .perturb import adaptive, both, cloak, occlude

__all__ = [
    "adaptive",
    "aurc",
    "bootstrap_ci",
    "both",
    "cloak",
    "compare",
    "coverage_at_risk",
    "evaluate_bundle",
    "expected_calibration_error",
    "fpr_at_coverage",
    "occlude",
    "paired_test",
    "selective_accuracy",
]
