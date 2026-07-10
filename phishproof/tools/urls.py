"""URL feature extraction shared by several verifiers."""

from __future__ import annotations

from .._release import pending


def registrable_domain(url: str) -> str:
    """Return the registrable (eTLD+1) domain of a URL."""
    pending("tools.urls.registrable_domain")


def url_features(url: str) -> dict[str, float]:
    """Return the lexical URL features used to cross-check host cues."""
    pending("tools.urls.url_features")
