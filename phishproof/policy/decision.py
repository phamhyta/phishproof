"""The act-or-abstain rule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..schema import Label, PageEvidence
from .frozen import FrozenPolicy
from .ranking import reaches


class Action(str, Enum):
    BLOCK = "block"
    CLEAR = "clear"
    ABSTAIN = "abstain"


class Reason(str, Enum):
    ACTED = "acted"
    INCOMPLETE_INFERENCE = "incomplete_inference"
    NO_GATING_CUE = "no_gating_cue"
    CHECK_UNAVAILABLE = "check_unavailable"
    CHECK_FAILED = "check_failed"
    BINDING_FAILED = "binding_failed"
    ACTION_DISABLED = "action_disabled"
    BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True)
class Decision:
    page_id: str
    action: Action
    reason: Reason
    verdict: Label | None
    rank: float | None
    correctness: float | None = None


def _gate_failure(evidence: PageEvidence) -> Reason | None:
    gating = evidence.gating_checks
    if not gating:
        return Reason.NO_GATING_CUE
    if not all(check.available for check in gating):
        return Reason.CHECK_UNAVAILABLE
    if not all(check.passed for check in gating):
        return Reason.CHECK_FAILED
    return None


def decide(evidence: PageEvidence, policy: FrozenPolicy) -> Decision:
    def abstain(reason: Reason) -> Decision:
        return Decision(evidence.page_id, Action.ABSTAIN, reason, evidence.verdict, evidence.rank)

    if not evidence.complete or evidence.verdict is None:
        return abstain(Reason.INCOMPLETE_INFERENCE)
    failure = _gate_failure(evidence)
    if failure is not None:
        return abstain(failure)

    if evidence.verdict is Label.PHISH:
        action, threshold = Action.BLOCK, policy.tau_block
    else:
        if not evidence.bound:
            return abstain(Reason.BINDING_FAILED)
        action, threshold = Action.CLEAR, policy.tau_clear

    if threshold is None:
        return abstain(Reason.ACTION_DISABLED)
    if not reaches(evidence.rank, threshold):
        return abstain(Reason.BELOW_THRESHOLD)
    return Decision(
        evidence.page_id,
        action,
        Reason.ACTED,
        evidence.verdict,
        evidence.rank,
        policy.correctness(evidence.rank),
    )
