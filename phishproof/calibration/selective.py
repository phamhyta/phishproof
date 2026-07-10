"""Selective decision rule: act above the calibrated threshold, else abstain."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..types import Decision


def decide(trust: float, tau: float) -> Decision:
    """Return ``act`` if calibrated trust ``>= tau``, else ``abstain``."""
    pending("calibration.selective.decide")


def choose_threshold(
    scores: Sequence[float],
    correct: Sequence[int],
    target_risk: float,
) -> float:
    """Pick ``tau`` on a held-out split to meet a target selective risk."""
    pending("calibration.selective.choose_threshold")


def risk_coverage(
    scores: Sequence[float],
    correct: Sequence[int],
) -> list[tuple[float, float]]:
    """Return the ``(coverage, selective-risk)`` operating curve."""
    pending("calibration.selective.risk_coverage")
