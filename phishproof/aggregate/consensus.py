"""Per-type agreement over the panel's cited cues.

The central signal: for each evidence type, how strongly the agents agree on a
single value, averaged uniformly across the active types (Section 4). The
counting logic is released with the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..schema import AgentOutput, CueType


def per_type_agreement(
    outputs: Sequence[AgentOutput],
    cue_types: Sequence[CueType],
) -> dict[CueType, float]:
    """Fraction of agents that concur on the majority value, per cue type.

    For type ``t``, ``a_t = (max agents citing one normalized value of t) / M``.
    Types absent from the page contribute ``0``. Returns one score per active
    type; the uniform mean of these is the GEA.
    """
    pending("aggregate.consensus.per_type_agreement")


def verdict_concurrence(outputs: Sequence[AgentOutput]) -> float:
    """Share of agents that share the majority label (a label-channel signal)."""
    pending("aggregate.consensus.verdict_concurrence")
