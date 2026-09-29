"""Paired bootstrap over pages for confidence intervals (experiments.tex §Statistical).

Numeric cells report mean +/- 95% CI from a paired bootstrap over pages: resample page
indices with replacement, recompute the metric on the resampled set, and take the
2.5/97.5 percentiles. "Paired" = the same resampled indices are used for every method, so
method differences are compared on identical page draws.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi). metric_fn takes an array of resampled page indices."""
    rng = np.random.default_rng(seed)
    point = metric_fn(np.arange(n))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = metric_fn(idx)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(point), lo, hi


def paired_bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Shared resample index matrix (n_boot x n) so every method uses identical draws."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def ci_from_draws(values_per_draw: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(values_per_draw, alpha / 2))
    hi = float(np.quantile(values_per_draw, 1 - alpha / 2))
    return lo, hi


def significant(ci_a: tuple[float, float], ci_b: tuple[float, float]) -> bool:
    """Non-overlapping 95% CIs => call the difference significant (paper's bar)."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return hi_a < lo_b or hi_b < lo_a
