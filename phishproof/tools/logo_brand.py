"""Logo verifier: match a rendered logo to a brand with a visual encoder."""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, GroundingResult
from ..types import PageRecord


class LogoBrandMatcher:
    """Ground a logo cue by matching the rendered logo against a brand library."""

    name = "logo"

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult:
        pending("tools.logo_brand.LogoBrandMatcher.verify")
