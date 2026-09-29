"""Structural DOM tools (deterministic, return 0/1).

Grounds two active cue types from the page HTML:
  - form_action_domain: the eTLD+1 a form posts to
  - credential_intent:  whether the page collects credentials
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import Cue, PageRecord
from .urls import registrable_domain, resolve_action

# Input types that signal credential collection.
_CRED_INPUT_TYPES = {"password"}
_CRED_NAME_HINTS = (
    "pass", "pwd", "passwd", "login", "signin", "user", "email", "card",
    "cvv", "ssn", "account", "pin", "otp",
)


@lru_cache(maxsize=2048)
def _load_soup(html_path: str) -> BeautifulSoup:
    html = Path(html_path).read_text(encoding="utf-8", errors="replace")
    return BeautifulSoup(html, "lxml")


def _soup(page: PageRecord) -> BeautifulSoup | None:
    if not page.dom_html_path or not Path(page.dom_html_path).exists():
        return None
    return _load_soup(page.dom_html_path)


def form_action_domains(page: PageRecord) -> set[str]:
    """Set of eTLD+1 domains that the page's forms post to."""
    soup = _soup(page)
    if soup is None:
        return set()
    base = page.final_url or page.url
    domains: set[str] = set()
    for form in soup.find_all("form"):
        action = form.get("action")
        resolved = resolve_action(base, action)
        dom = registrable_domain(resolved)
        if dom:
            domains.add(dom)
    # Forms with no action post to the page itself.
    if not domains:
        self_dom = registrable_domain(base)
        if self_dom and soup.find("form"):
            domains.add(self_dom)
    return domains


def has_credential_intent(page: PageRecord) -> bool:
    """True if the page appears to collect credentials (password field or login form)."""
    soup = _soup(page)
    if soup is None:
        return False
    for inp in soup.find_all("input"):
        itype = (inp.get("type") or "").lower()
        if itype in _CRED_INPUT_TYPES:
            return True
        name = " ".join(
            str(inp.get(a, "")) for a in ("name", "id", "placeholder", "autocomplete")
        ).lower()
        if itype in ("text", "email") and any(h in name for h in _CRED_NAME_HINTS):
            return True
    return False


# --- Grounding functions: re-derive the cue and score the agent's claim in [0,1] ---

def verify_form_action_domain(cue: Cue, page: PageRecord) -> float:
    """1.0 if the claimed eTLD+1 is actually a form-action domain on the page."""
    claimed = (cue.value or "").lower()
    return 1.0 if claimed and claimed in form_action_domains(page) else 0.0


def _as_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in ("yes", "true", "1", "phish", "present"):
        return True
    if v in ("no", "false", "0", "absent", "none"):
        return False
    return None


def verify_credential_intent(cue: Cue, page: PageRecord) -> float:
    """1.0 if the agent's yes/no credential-intent claim matches the DOM."""
    claim = _as_bool(cue.value)
    if claim is None:
        return 0.0
    return 1.0 if claim == has_credential_intent(page) else 0.0
