"""Registrable-domain handling, using the Public Suffix List shipped with tldextract."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

import tldextract

# Use the bundled suffix list (no network at runtime); refresh offline if needed.
_extract = tldextract.TLDExtract(suffix_list_urls=())


def registrable_domain(url_or_host: str | None) -> str | None:
    """eTLD+1 of a URL or bare host, e.g. 'paypal-secure.com/login' -> 'paypal-secure.com'.

    Returns None for empty / non-resolvable inputs (e.g. 'javascript:', '#', '').
    """
    if not url_or_host:
        return None
    s = url_or_host.strip()
    if not s:
        return None
    try:
        ext = _extract(s)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}".lower()
        # No suffix (IP address, localhost, relative path) -> fall back to netloc/host.
        netloc = urlparse(s if "//" in s else "//" + s).netloc
        return netloc.lower() or None
    except ValueError:
        # malformed URL (e.g. bad IPv6 brackets in a page's href) -> not resolvable
        return None


def resolve_action(base_url: str | None, action: str | None) -> str | None:
    """Resolve a possibly-relative form action against the page URL -> absolute URL."""
    if action is None:
        return base_url
    action = action.strip()
    if not action or action.lower().startswith(("javascript:", "mailto:", "#", "data:")):
        return base_url  # self-post / no real target
    if base_url:
        return urljoin(base_url, action)
    return action


def registrable_label(url_or_host: str | None) -> str | None:
    """Registrable domain without its public suffix: 'https://a.paypal.co.uk/x' -> 'paypal'."""
    if not url_or_host or not url_or_host.strip():
        return None
    try:
        return _extract(url_or_host.strip()).domain.lower() or None
    except ValueError:
        return None


def hostname(url_or_host: str | None) -> str | None:
    """Full lowercase host of a URL or bare host, or None if there is none."""
    if not url_or_host or not url_or_host.strip():
        return None
    text = url_or_host.strip()
    try:
        return urlparse(text if "//" in text else "//" + text).hostname
    except ValueError:
        return None
