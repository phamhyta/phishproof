"""Brand-host binding, which a clear requires in addition to the checks on cited cues.

The binding holds when every brand the panel names equals the registrable label of the page's
host. A brand on the reference list must also be served from one of its listed domains.
"""

from __future__ import annotations

import json
import pickle
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..schema import BRAND_CUE_TYPES, MemberOutput
from .brands import canonical_brand
from .urls import registrable_domain, registrable_label


def squeeze(name: str) -> str:
    """Lowercase a name and remove its spaces and punctuation."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class BrandDomains:
    """Reference list mapping a brand to its legitimate registrable domains."""

    def __init__(self, domains: Mapping[str, Iterable[str]] | None = None) -> None:
        self._domains: dict[str, set[str]] = {}
        for brand, listed in (domains or {}).items():
            key = squeeze(canonical_brand(brand) or brand)
            self._domains.setdefault(key, set()).update(
                registrable_domain(domain) or domain.lower() for domain in listed
            )

    @classmethod
    def from_json(cls, path: str | Path) -> BrandDomains:
        return cls(json.loads(Path(path).read_text()))

    @classmethod
    def from_pickle(cls, path: str | Path) -> BrandDomains:
        """Read ``domain_map.pkl`` as released with Phishpedia. Only load files you trust."""
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def get(self, brand: str) -> set[str] | None:
        return self._domains.get(squeeze(brand))


def named_brands(outputs: list[MemberOutput]) -> set[str]:
    """Normalized brands that panel members name in their brand and logo cues."""
    brands = {
        canonical_brand(cue.value)
        for output in outputs
        for cue in output.cues
        if cue.type in BRAND_CUE_TYPES
    }
    return {brand for brand in brands if brand}


def binding_holds(brands: Iterable[str], url: str, reference: BrandDomains | None = None) -> bool:
    brands = list(brands)
    label = squeeze(registrable_label(url) or "")
    domain = registrable_domain(url)
    if not brands or not label or not domain:
        return False
    for brand in brands:
        if squeeze(brand) != label:
            return False
        listed = reference.get(brand) if reference else None
        if listed is not None and domain not in listed:
            return False
    return True
