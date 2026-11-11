"""Domain handling of the re-implemented detectors, and the shared brand normalization."""

from __future__ import annotations

import tldextract

from ...verification.binding import squeeze
from ...verification.brands import canonical_brand

# the full Public Suffix List including its PRIVATE section, read from tldextract's bundled snapshot with no network, so evil.github.io is its own registrable domain.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

def registrable_domain(url_or_host: str) -> str:
    """Return the registrable domain of a URL or host, or "" for IP addresses and unparsable input."""
    parts = _EXTRACT(url_or_host.strip())
    if not parts.domain or not parts.suffix:
        return ""
    return f"{parts.domain}.{parts.suffix}".lower()


def registrable_label(url_or_host: str) -> str:
    """Return the registrable domain without its public suffix, or "" when there is none."""
    parts = _EXTRACT(url_or_host.strip())
    if not parts.domain or not parts.suffix:
        return ""
    return parts.domain.lower()


def normalize_brand(name: str) -> str:
    return canonical_brand(name) or ""


def sq(text: str) -> str:
    return squeeze(text)
