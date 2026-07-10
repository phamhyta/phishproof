"""Isotonic calibration of the trust score to P(correct)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .._release import pending


class IsotonicCalibrator:
    """Monotonic map from the raw trust score to a correctness probability.

    Fit on a held-out split so the reported probability can be acted on
    directly. The fitted knots serialize to and from JSON.
    """

    def fit(self, scores: Sequence[float], correct: Sequence[int]) -> "IsotonicCalibrator":
        pending("calibration.isotonic.IsotonicCalibrator.fit")

    def predict(self, scores: Sequence[float]) -> list[float]:
        pending("calibration.isotonic.IsotonicCalibrator.predict")

    def save(self, path: str | Path) -> None:
        pending("calibration.isotonic.IsotonicCalibrator.save")

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        pending("calibration.isotonic.IsotonicCalibrator.load")
