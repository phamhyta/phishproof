"""Redirect-target check. Unavailable when the capture has no redirect chain."""

from __future__ import annotations

from ..schema import Cue, PageRecord
from .urls import registrable_domain


def verify_redirect_target(cue: Cue, page: PageRecord) -> float | None:
    if not page.redirect_chain:
        return None
    final = registrable_domain(page.redirect_chain[-1])
    claimed = (cue.value or "").lower()
    return 1.0 if claimed and final and claimed == final else 0.0
