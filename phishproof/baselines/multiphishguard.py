"""MultiPhishGuard-style consolidated baseline (B6)."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..schema import AgentOutput


class MultiPhishGuard:
    """Consolidator that returns high confidence whenever the agents agree.

    Reproduced faithfully as the strongest ensemble baseline; unlike PhishProof
    it does not check whether the agreed reasons hold on the page.
    """

    def score(self, outputs: Sequence[AgentOutput]) -> float:
        pending("baselines.multiphishguard.MultiPhishGuard.score")
