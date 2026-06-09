"""URL / domain normalization. eTLD+1 via the public suffix list (tldextract)."""

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
