"""Reliability scores computed directly from the panel's labels, confidences and cues."""

from __future__ import annotations

from collections import Counter

from ..policy.ranking import mean_confidence
from ..schema import Label, MemberOutput

CONCURRENCE_WEIGHT = 1e-3


def _member(outputs: list[MemberOutput], member_id: str) -> MemberOutput | None:
    return next((output for output in outputs if output.member_id == member_id), None)


def _votes(outputs: list[MemberOutput]) -> Counter[Label]:
    return Counter(output.verdict for output in outputs if output.verdict is not None)


def vision_confidence(outputs: list[MemberOutput], vision_id: str) -> float | None:
    """B1, the vision model's self-reported confidence."""
    member = _member(outputs, vision_id)
    return member.confidence if member else None


def panel_mean_confidence(outputs: list[MemberOutput]) -> float | None:
    """B2, the mean of the available panel confidences, which is the deployed ranker."""
    return mean_confidence(outputs)


def majority_share(outputs: list[MemberOutput]) -> float:
    """B3, the share of the panel that gave the majority label."""
    votes = _votes(outputs)
    return votes.most_common(1)[0][1] / len(outputs) if votes else 0.0


def majority_confidence(outputs: list[MemberOutput]) -> float:
    """B5-P, the mean confidence of the members in the majority."""
    votes = _votes(outputs)
    if not votes:
        return 0.0
    majority = votes.most_common(1)[0][0]
    agreeing = [output for output in outputs if output.verdict is majority]
    confidences = [output.confidence for output in agreeing if output.confidence is not None]
    return sum(confidences) / len(confidences) if confidences else len(agreeing) / len(outputs)


def vision_with_concurrence(outputs: list[MemberOutput], vision_id: str) -> float | None:
    """The alternative score s, vision confidence with a text-concurrence tie-break."""
    vision = _member(outputs, vision_id)
    if vision is None or vision.confidence is None:
        return None
    text = [output for output in outputs if output.member_id != vision_id]
    agreeing = sum(output.verdict is vision.verdict for output in text)
    concurrence = agreeing / len(text) if text else 0.0
    return vision.confidence + CONCURRENCE_WEIGHT * concurrence
