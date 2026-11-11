"""Paired bootstrap, resampling pages or duplicate groups with the same draws for every method."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_indices(n: int, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def group_bootstrap_indices(
    groups: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> list[np.ndarray]:
    """Resample whole duplicate groups, because near-duplicate pages are not independent."""
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([members[group] for group in rng.integers(0, len(unique), len(unique))])
        for _ in range(n_boot)
    ]


def ci_from_draws(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
