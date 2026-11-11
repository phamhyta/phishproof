"""Brand check: does the page's own identity text present the claimed brand?"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import Cue, PageRecord
from .brands import canonical_brand

_IDENTITY_META = ("og:site_name", "og:title", "application-name", "author")
_LEADING_TEXT_CHARS = 1500


def _identity_text(page: PageRecord) -> str:
    html = Path(page.dom_html_path).read_text(errors="replace")
    soup = BeautifulSoup(html, "lxml")
    parts: list[str] = [soup.title.string] if soup.title and soup.title.string else []
    for meta in soup.find_all("meta"):
        if (meta.get("property") or meta.get("name") or "").lower() in _IDENTITY_META:
            parts.append(meta.get("content") or "")
    parts.extend(heading.get_text(" ") for heading in soup.find_all(["h1", "h2"])[:3])
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    parts.append(soup.get_text(" ")[:_LEADING_TEXT_CHARS])
    return " ".join(parts).lower()


def verify_brand_claim(cue: Cue, page: PageRecord) -> float:
    claimed = canonical_brand(cue.value)
    if not claimed:
        return 0.0
    text = _identity_text(page)
    tokens = [token for token in claimed.split() if len(token) >= 3] or [claimed]
    return 1.0 if any(token in text for token in tokens) else 0.0
