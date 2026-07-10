"""Evidence-targeted and white-box perturbations for the fail-safe study.

Each perturbation corrupts the cues a method relies on so we can measure whether
the detector abstains instead of emitting a confident wrong label.
"""

from __future__ import annotations

from .._release import pending
from ..types import PageRecord


def cloak(page: PageRecord) -> PageRecord:
    """Corrupt the structural cues (form action, brand strings)."""
    pending("eval.perturb.cloak")


def occlude(page: PageRecord) -> PageRecord:
    """Corrupt the rendered logo the vision agent relies on."""
    pending("eval.perturb.occlude")


def both(page: PageRecord) -> PageRecord:
    """Apply ``cloak`` and ``occlude`` together."""
    pending("eval.perturb.both")


def adaptive(page: PageRecord) -> PageRecord:
    """White-box attack that targets every cited verifier at once."""
    pending("eval.perturb.adaptive")
