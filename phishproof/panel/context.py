"""Compact page context shown to every panel member."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import PageRecord
from ..verification.dom import form_action_domains, has_credential_intent
from ..verification.urls import registrable_domain

# The same small limits apply to every page, so that DOM size cannot act as a class signal.
_MAX_TEXT = 500
_MAX_LINKS = 5
_MAX_CONTEXT_CHARS = 7000


def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:_MAX_TEXT]


def render_context(page: PageRecord) -> str:
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
