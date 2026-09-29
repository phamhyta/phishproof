"""Certificate-org grounding (structural, 0/1).

INACTIVE on the Phishpedia dataset (no TLS cert captured) — returns None, which the
registry treats as N/A and drops from the consensus, per decision 0b. Implemented so a
backfill capture that fills page.certificate_org re-enables the cue with no code change.
"""

from __future__ import annotations

from ..schema import Cue, PageRecord


def _canon(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def verify_certificate_org(cue: Cue, page: PageRecord) -> float | None:
    if page.certificate_org is None:
        return None  # N/A — not captured for this page
    claimed = _canon(cue.value)
    actual = _canon(page.certificate_org)
    if not claimed or not actual:
        return 0.0
    return 1.0 if claimed in actual or actual in claimed else 0.0
