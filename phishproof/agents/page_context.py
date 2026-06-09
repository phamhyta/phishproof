"""Render a compact textual context for a page so agents spend few tokens on the DOM.

Extracts the signal-bearing parts (title, forms + action domains, input fields, link
domains, a visible-text snippet) instead of dumping raw HTML. The vision agent
additionally receives the screenshot as an image (see client.py).
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import PageRecord
from ..tools.dom import form_action_domains, has_credential_intent
from ..tools.urls import registrable_domain

# Kept small and EQUAL for both classes so the agents judge legitimacy, not page
# complexity: benign homepages have ~20x more links / ~7x larger DOM than the minimal
# phishing pages, and revealing that would be a spurious shortcut (see data audit).
_MAX_TEXT = 500
_MAX_LINKS = 5
_MAX_CONTEXT_CHARS = 7000   # hard cap so the raw DOM size does not leak through


def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:_MAX_TEXT]


def render_context(page: PageRecord) -> str:
    """Compact text the agent reads. Page URL host is given; cue values must be earned."""
    lines: list[str] = [f"URL: {page.url}", f"URL_host_etld1: {registrable_domain(page.url)}"]

    soup = None
    if page.dom_html_path and Path(page.dom_html_path).exists():
        html = Path(page.dom_html_path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

    if soup is not None:
        title = soup.title.string if soup.title and soup.title.string else ""
        lines.append(f"Title: {title.strip()[:160]}")

        fads = sorted(form_action_domains(page))
        lines.append(f"Form action domains (eTLD+1): {fads or 'none'}")

        input_types: list[str] = []
        for inp in soup.find_all("input"):
            t = (inp.get("type") or "text").lower()
            nm = (inp.get("name") or inp.get("id") or "").lower()
            input_types.append(f"{t}:{nm}" if nm else t)
        lines.append(f"Input fields: {input_types[:20] or 'none'}")
        lines.append(f"Has password field / credential form: {has_credential_intent(page)}")

        link_domains: list[str] = []
        for a in soup.find_all("a", href=True):
            d = registrable_domain(a["href"])
            if d and d not in link_domains:
                link_domains.append(d)
        lines.append(f"Outbound link domains: {link_domains[:_MAX_LINKS] or 'none'}")

        lines.append(f"Visible text (truncated): {_visible_text(soup)}")
    else:
        lines.append("(no DOM captured)")

    return "\n".join(lines)[:_MAX_CONTEXT_CHARS]
