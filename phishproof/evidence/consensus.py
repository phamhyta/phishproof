"""Which cues a strict majority of the panel asserts."""

from __future__ import annotations

from collections import defaultdict

from ..schema import Cue, MemberOutput
from .normalize import normalize_cue

CueKey = tuple[str, str]


def normalized_cues(outputs: list[MemberOutput]) -> dict[CueKey, Cue]:
    """Distinct normalized cues over the panel, one representative for each."""
    cues: dict[CueKey, Cue] = {}
    for output in outputs:
        for cue in output.cues:
            normalized = normalize_cue(cue)
            if normalized is not None:
                cues.setdefault(normalized.key(), normalized)
    return cues


def cue_supporters(outputs: list[MemberOutput]) -> dict[CueKey, set[str]]:
    supporters: dict[CueKey, set[str]] = defaultdict(set)
    for output in outputs:
        for cue in output.cues:
            normalized = normalize_cue(cue)
            if normalized is not None:
                supporters[normalized.key()].add(output.member_id)
    return supporters


def consensus_cues(outputs: list[MemberOutput]) -> list[tuple[Cue, set[str]]]:
    """Cues asserted by more than half of the panel, each with the members that assert it."""
    supporters = cue_supporters(outputs)
    majority = len(outputs) / 2
    return [
        (cue, supporters[key])
        for key, cue in normalized_cues(outputs).items()
        if len(supporters[key]) > majority
    ]
