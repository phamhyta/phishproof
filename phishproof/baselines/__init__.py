"""Reliability baselines and dedicated detectors compared with the proposed policy."""

from .combiner import ConfidenceAgreementCombiner
from .consolidator import Consolidator
from .dawid_skene import dawid_skene
from .registry import PANEL_BASELINES, compute_panel_baselines
from .scores import (
    majority_confidence,
    majority_share,
    panel_mean_confidence,
    vision_confidence,
    vision_with_concurrence,
)

__all__ = [
    "PANEL_BASELINES",
    "ConfidenceAgreementCombiner",
    "Consolidator",
    "compute_panel_baselines",
    "dawid_skene",
    "majority_confidence",
    "majority_share",
    "panel_mean_confidence",
    "vision_confidence",
    "vision_with_concurrence",
]
