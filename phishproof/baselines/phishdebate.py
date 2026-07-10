"""PhishDebate-style moderated baseline (B7)."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..schema import AgentOutput


class PhishDebate:
    """Moderated debate that resolves disagreement into a single confidence.

    Converts doubt into confidence through a moderator, in contrast to
    PhishProof, which preserves disagreement as the abstain signal.
    """

    def score(self, outputs: Sequence[AgentOutput]) -> float:
        pending("baselines.phishdebate.PhishDebate.score")
