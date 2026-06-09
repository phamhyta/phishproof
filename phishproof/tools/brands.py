"""Brand value normalization -> canonical lexicon entry (used by C2 and grounding)."""

from __future__ import annotations

import re

# Corporate suffixes / noise to strip so 'PayPal Inc.' and 'PayPal' canonicalize equal.
_SUFFIXES = (
    "incorporated", "inc", "corporation", "corp", "company", "co", "ltd",
    "limited", "llc", "plc", "group", "sa", "ag", "nv", "gmbh", "holdings",
)
_SUFFIX_RE = re.compile(r"\b(" + "|".join(_SUFFIXES) + r")\b", re.IGNORECASE)


def canonical_brand(brand: str | None) -> str | None:
    """'Facebook, Inc.' -> 'facebook'; 'Microsoft OneDrive' -> 'microsoft onedrive'."""
    if not brand:
        return None
    s = brand.lower()
    s = _SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)  # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s or None
