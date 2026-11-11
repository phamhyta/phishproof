"""Certificate-organization check. Unavailable when the capture has no certificate."""

from __future__ import annotations

from ..schema import Cue, PageRecord


def _canon(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def verify_certificate_org(cue: Cue, page: PageRecord) -> float | None:
    if page.certificate_org is None:
        return None
    claimed = _canon(cue.value)
    actual = _canon(page.certificate_org)
    if not claimed or not actual:
        return 0.0
    return 1.0 if claimed in actual or actual in claimed else 0.0
