"""Aggregation: per-type agreement and the GEA trust signal."""

from .consensus import per_type_agreement, verdict_concurrence
from .gea import gea_score, score_page, trust_signal
from .normalize import normalize_cue, normalize_value

__all__ = [
    "gea_score",
    "normalize_cue",
    "normalize_value",
    "per_type_agreement",
    "score_page",
    "trust_signal",
    "verdict_concurrence",
]
