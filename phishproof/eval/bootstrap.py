"""Bootstrap confidence intervals and paired significance tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .._release import pending


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic."""
    pending("eval.bootstrap.bootstrap_ci")


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 1000,
) -> float:
    """Two-sided paired bootstrap p-value for a difference of statistics."""
    pending("eval.bootstrap.paired_test")
