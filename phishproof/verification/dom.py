"""DOM checks for form-action and credential cues. Each returns 0 or 1."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import Cue, PageRecord
from .urls import registrable_domain, resolve_action

_NAMED_INPUT_TYPES = ("text", "email")
_CREDENTIAL_ATTRIBUTES = ("name", "id", "placeholder", "autocomplete")
_CREDENTIAL_HINTS = (
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
    """Registrable domains that the page's forms post to."""
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
    """True for a password input, or a text or e-mail input named as a credential."""
    soup = _soup(page)
    if soup is None:
        return False
    for field in soup.find_all("input"):
        field_type = (field.get("type") or "").lower()
        if field_type == "password":
            return True
        attributes = " ".join(str(field.get(name, "")) for name in _CREDENTIAL_ATTRIBUTES).lower()
        if field_type in _NAMED_INPUT_TYPES and any(h in attributes for h in _CREDENTIAL_HINTS):
            return True
    return False


def verify_form_action_domain(cue: Cue, page: PageRecord) -> float:
    """1.0 if the cited domain is a form-action domain of the page."""
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
    """1.0 if the cited yes/no value matches the DOM."""
    claim = _as_bool(cue.value)
    if claim is None:
        return 0.0
    return 1.0 if claim == has_credential_intent(page) else 0.0
