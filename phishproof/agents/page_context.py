"""Assemble the per-page context handed to each agent."""

from __future__ import annotations

from dataclasses import dataclass

from .._release import pending
from ..types import PageRecord


@dataclass
class PageContext:
    """Rendered and structural views of one page.

    Bundles the screenshot, the parsed DOM text, and the URL so every agent sees
    a consistent snapshot of the page under inspection.
    """

    page: PageRecord
    dom_text: str = ""
    screenshot_path: str | None = None


def build_context(page: PageRecord) -> PageContext:
    """Load and normalize the page's screenshot and DOM into a PageContext."""
    pending("agents.page_context.build_context")
