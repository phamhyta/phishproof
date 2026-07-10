"""DOM verifier: extract and check where credential forms post."""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, GroundingResult
from ..types import PageRecord


class FormActionVerifier:
    """Ground a form-action cue by parsing the page's form targets."""

    name = "form_action"

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult:
        pending("tools.dom.FormActionVerifier.verify")


def extract_form_actions(html_path: str) -> list[str]:
    """Return the action targets of every credential form on the page."""
    pending("tools.dom.extract_form_actions")
