"""Run the full selective-detection evaluation for one corpus."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .._release import pending


def evaluate_bundle(bundle_path: str | Path) -> dict[str, float]:
    """Compute AURC / SelAcc / FPR / Cov99 / ECE for a scored result bundle."""
    pending("eval.evaluate.evaluate_bundle")


def compare(
    method_scores: Sequence[float],
    baseline_scores: Sequence[float],
    correct: Sequence[int],
) -> dict[str, float]:
    """Paired comparison (delta, bootstrap CI, p-value) between two methods."""
    pending("eval.evaluate.compare")
