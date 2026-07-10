"""Value normalization for evidence cues.

Two agents agree only if their cited values match after normalization (for
example, registrable-domain canonicalization of a form action). The rules are
released with the paper.
"""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, CueType


def normalize_value(cue_type: CueType, value: str) -> str:
    """Canonicalize a cue value so that equal claims compare equal."""
    pending("aggregate.normalize.normalize_value")


def normalize_cue(cue: Cue) -> Cue:
    """Return a copy of a cue with its value normalized."""
    pending("aggregate.normalize.normalize_cue")
