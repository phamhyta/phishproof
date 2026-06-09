"""Calibration + selective decision (C4)."""

from .isotonic import IsotonicCalibrator
from .per_class import PerClassCalibrator
from .selective import (
    apply_selection,
    correctness,
    labels_from_pages,
    risk_coverage,
    threshold_for_coverage,
    threshold_for_target_risk,
)

__all__ = [
    "IsotonicCalibrator",
    "PerClassCalibrator",
    "correctness",
    "risk_coverage",
    "threshold_for_target_risk",
    "threshold_for_coverage",
    "apply_selection",
    "labels_from_pages",
]
