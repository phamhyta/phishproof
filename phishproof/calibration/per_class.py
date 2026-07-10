"""Per-class calibration variant (studied in the ablations)."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..types import Label


class PerClassCalibrator:
    """Fit a separate calibrator per predicted label."""

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[Label],
        correct: Sequence[int],
    ) -> "PerClassCalibrator":
        pending("calibration.per_class.PerClassCalibrator.fit")

    def predict(self, scores: Sequence[float], labels: Sequence[Label]) -> list[float]:
        pending("calibration.per_class.PerClassCalibrator.predict")
