"""Grounded Evidence-Agreement, reported for every page as a diagnostic.

GEA does not rank pages and adds no acceptance condition to the decision policy.
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import ACTIVE_CUE_TYPES, CueType, MemberOutput
from .normalize import normalize_cue


def per_type_agreement(outputs: list[MemberOutput]) -> dict[CueType, float]:
    """For each active type, the largest share of the panel citing one value of that type."""
    if not outputs:
        return {}
    agreement: dict[CueType, float] = {}
    for cue_type in ACTIVE_CUE_TYPES:
        members_by_value: dict[str, set[str]] = defaultdict(set)
        for output in outputs:
            for cue in output.cues:
                if cue.type is cue_type and (normalized := normalize_cue(cue)) is not None:
                    members_by_value[normalized.value].add(output.member_id)
        largest = max((len(members) for members in members_by_value.values()), default=0)
        agreement[cue_type] = largest / len(outputs)
    return agreement


def grounded_evidence_agreement(outputs: list[MemberOutput]) -> float:
    agreement = per_type_agreement(outputs)
    return sum(agreement.values()) / len(ACTIVE_CUE_TYPES) if agreement else 0.0
