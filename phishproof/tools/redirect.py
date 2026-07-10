"""Redirect-chain verifier."""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, GroundingResult
from ..types import PageRecord


class RedirectVerifier:
    """Ground a redirect cue by resolving the page's redirect chain."""

    name = "redirect"

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult:
        pending("tools.redirect.RedirectVerifier.verify")
