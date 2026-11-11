"""Baseline scores that can be computed from stored panel outputs alone."""

from __future__ import annotations

from ..schema import MemberOutput
from .dawid_skene import dawid_skene
from .scores import majority_confidence, majority_share, panel_mean_confidence, vision_confidence

PANEL_BASELINES = ("B1", "B2", "B3", "B4", "B5-P")


def compute_panel_baselines(
    outputs_by_page: dict[str, list[MemberOutput]], vision_id: str
) -> dict[str, dict[str, float | None]]:
    """Return {baseline: {page_id: score}}. B4 is fitted over all the pages it is given."""
    return {
        "B1": {p: vision_confidence(o, vision_id) for p, o in outputs_by_page.items()},
        "B2": {p: panel_mean_confidence(o) for p, o in outputs_by_page.items()},
        "B3": {p: majority_share(o) for p, o in outputs_by_page.items()},
        "B4": dawid_skene(outputs_by_page),
        "B5-P": {p: majority_confidence(o) for p, o in outputs_by_page.items()},
    }
