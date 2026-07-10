"""Label-channel reliability baselines.

Reliability read from the label distribution: single-model confidence, majority
vote, and weighted label agreement. Released with the paper.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..schema import AgentOutput


def single_model_confidence(outputs: Sequence[AgentOutput]) -> float:
    """B1: one model's reported class probability."""
    pending("baselines.label_scores.single_model_confidence")


def majority_vote(outputs: Sequence[AgentOutput]) -> float:
    """B4: fraction of agents on the majority label."""
    pending("baselines.label_scores.majority_vote")


def weighted_label_agreement(outputs: Sequence[AgentOutput]) -> float:
    """B5: label agreement weighted by per-model reliability (Dawid-Skene)."""
    pending("baselines.label_scores.weighted_label_agreement")
