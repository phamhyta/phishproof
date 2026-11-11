"""Verify the panel's consensus cues and keep a record of every check."""

from __future__ import annotations

from ..evidence import consensus_cues
from ..schema import PERCEPTUAL_CUE_TYPES, CueCheck, MemberOutput, PageRecord
from ..verification import Verifiers, verify_cue

STRUCTURAL_PASS_SCORE = 1.0


def check_consensus(
    outputs: list[MemberOutput],
    page: PageRecord,
    logo_threshold: float,
    verifiers: Verifiers | None = None,
) -> list[CueCheck]:
    checks: list[CueCheck] = []
    for cue, supporters in consensus_cues(outputs):
        threshold = logo_threshold if cue.type in PERCEPTUAL_CUE_TYPES else STRUCTURAL_PASS_SCORE
        result = verify_cue(cue, page, verifiers)
        checks.append(
            CueCheck(
                type=cue.type,
                value=cue.value,
                supporters=sorted(supporters),
                tool=result.tool if result else None,
                score=result.score if result else None,
                threshold=threshold,
                passed=result.score >= threshold if result else None,
            )
        )
    return checks


def groundedness(checks: list[CueCheck]) -> float:
    """Mean score of the checks that returned a value, reported as a diagnostic."""
    scores = [check.score for check in checks if check.available]
    return sum(scores) / len(scores) if scores else 0.0
