"""Brand-claim grounding via a specialized detector used as a tool (D1/D2/D3).

The brand cue is grounded by re-deriving the page's brand with an existing detector
(method.tex §sec:ground): a reference-based visual matcher (Phishpedia/PhishIntention)
or an LLM brand-and-domain checker (PhishLLM). The real detectors are wired in Phase 5;
this module defines the interface plus a gold stand-in so the pipeline is testable now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schema import Cue, PageRecord
from .brands import canonical_brand


class BrandDetector(Protocol):
    """Re-derives the brand a page poses as, or None if it cannot."""

    name: str

    def derive_brand(self, page: PageRecord) -> str | None: ...


class GoldBrandDetector:
    """Stand-in that returns the dataset's gold brand annotation.

    Lets us exercise the grounding pipeline before the real D1/D2/D3 are wired.
    NOTE: do not use for the final RQ numbers — grounding against the gold label is
    circular; swap in PhishpediaDetector / PhishLLMDetector in Phase 5.
    """

    name = "gold"

    def derive_brand(self, page: PageRecord) -> str | None:
        return canonical_brand(page.brand)


class HtmlBrandDetector:
    """Non-circular brand grounding: does the PAGE itself present the claimed brand?

    Checks the page's textual identity (title, og:site_name, headings, leading visible
    text) for the claimed brand. Symmetric for phish and benign (uses page content, not
    the gold label), so it grounds a brand cue iff the page actually shows that brand —
    a PayPal phish does present PayPal branding, and so does the real paypal.com.
    A lightweight stand-in for the reference-based detectors (D1/D2/D3), wired in Phase 5.
    """

    name = "html"

    def __init__(self, head_chars: int = 1500) -> None:
        self.head_chars = head_chars

    def _identity_text(self, page: PageRecord) -> str:
        if not page.dom_html_path or not Path(page.dom_html_path).exists():
            return ""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            Path(page.dom_html_path).read_text(errors="replace"), "lxml"
        )
        parts: list[str] = []
        if soup.title and soup.title.string:
            parts.append(soup.title.string)
        for m in soup.find_all("meta"):
            if (m.get("property") or m.get("name") or "").lower() in (
                "og:site_name", "og:title", "application-name", "author",
            ):
                parts.append(m.get("content") or "")
        for h in soup.find_all(["h1", "h2"])[:3]:
            parts.append(h.get_text(" "))
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts.append(soup.get_text(" ")[: self.head_chars])
        return " ".join(parts).lower()

    def derive_brand(self, page: PageRecord) -> str | None:  # kept for the protocol
        return canonical_brand(page.brand)

    def grounds(self, claimed_brand: str, page: PageRecord) -> float:
        claimed = canonical_brand(claimed_brand)
        if not claimed:
            return 0.0
        text = self._identity_text(page)
        if not text:
            return 0.0
        # A brand grounds if any of its significant tokens appears in the page identity.
        tokens = [t for t in claimed.split() if len(t) >= 3]
        if not tokens:
            tokens = [claimed]
        return 1.0 if any(t in text for t in tokens) else 0.0


def verify_brand_claim(
    cue: Cue, page: PageRecord, detector: BrandDetector | None = None
) -> float | None:
    """1.0 if the page actually presents the claimed brand, else 0.0.

    Default detector derives the brand from page content (HtmlBrandDetector). If a
    gold-style detector is passed, fall back to canonical-equality against its brand.
    """
    if detector is None:
        return HtmlBrandDetector().grounds(cue.value, page)
    if isinstance(detector, HtmlBrandDetector):
        return detector.grounds(cue.value, page)
    derived = detector.derive_brand(page)
    if derived is None:
        return None
    claimed = canonical_brand(cue.value)
    if not claimed:
        return 0.0
    return 1.0 if claimed == derived else 0.0
