"""Aggregation: normalize -> consensus -> agreement (A) -> grounding (G) -> GEA (C2, C3)."""

from .consensus import agreement, consensus_cues, shared_cue_set
from .gea import groundedness, score_page
from .normalize import normalize_cue, normalize_value

__all__ = [
    "normalize_cue",
    "normalize_value",
    "shared_cue_set",
    "consensus_cues",
    "agreement",
    "groundedness",
    "score_page",
]
