"""Redirect-target grounding (structural, 0/1).

INACTIVE on the Phishpedia dataset (no redirect chain captured) — returns None (N/A),
per decision 0b. Re-enabled automatically when page.redirect_chain is backfilled.
"""

from __future__ import annotations

from ..schema import Cue, PageRecord
from .urls import registrable_domain


def verify_redirect_target(cue: Cue, page: PageRecord) -> float | None:
    if not page.redirect_chain:
        return None  # N/A — not captured
    final = registrable_domain(page.redirect_chain[-1])
    claimed = (cue.value or "").lower()
    return 1.0 if claimed and final and claimed == final else 0.0
