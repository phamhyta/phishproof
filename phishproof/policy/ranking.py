"""The ranking score: the plain mean of the confidences the panel returned."""

from __future__ import annotations

from ..schema import MemberOutput

_SCORE_TOLERANCE = 1e-9


def mean_confidence(outputs: list[MemberOutput]) -> float | None:
    confidences = [output.confidence for output in outputs if output.confidence is not None]
    return sum(confidences) / len(confidences) if confidences else None


def reaches(score: float | None, level: float | None) -> bool:
    """Whether a score reaches a threshold, ignoring floating-point noise in the mean."""
    return score is not None and level is not None and score + _SCORE_TOLERANCE >= level
