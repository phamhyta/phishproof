"""Reliability baselines B1-B6 (experiments.tex setup)."""

from .dawid_skene import dawid_skene
from .label_scores import (
    b1_single_model_confidence,
    b2_verbalized_confidence,
    b4_majority_vote,
)
from .multiphishguard import b6_multiphishguard_proxy
from .registry import PANEL_BASELINES, compute_panel_baselines
from .selfconsistency import self_consistency_label, self_consistency_score

__all__ = [
    "b1_single_model_confidence",
    "b2_verbalized_confidence",
    "b4_majority_vote",
    "b6_multiphishguard_proxy",
    "dawid_skene",
    "self_consistency_score",
    "self_consistency_label",
    "compute_panel_baselines",
    "PANEL_BASELINES",
]
