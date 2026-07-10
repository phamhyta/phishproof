"""HTML brand verifier: does the page belong to the brand it claims?"""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, GroundingResult
from ..types import PageRecord


class HtmlBrandDetector:
    """Re-derive the impersonated brand from page content and compare to host."""

    name = "brand"

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult:
        pending("tools.brands.HtmlBrandDetector.verify")
