"""Brand-domain consistency — a deterministic structural cue computed for every page.

Addresses the benign asymmetry: the rest of the cue schema is phishing-oriented, so a
legitimate page (a brand on its own domain) raises few cues and gets a near-zero GEA,
sinking it below the abstain threshold. This cue gives BOTH classes a positive, grounded
piece of evidence — "consistent" when the page's brand matches its own domain (benign),
"inconsistent" when a brand is shown on an unrelated domain (the phishing signal).

It is computed by a tool (not asserted by the agents), so it is always in the consensus
and always grounds to 1.0 — like an always-on structural check (cf. cert / redirect).
"""

from __future__ import annotations

from ..schema import Cue, CueType, PageRecord
from .brands import canonical_brand
from .urls import registrable_domain

_STOP = {"com", "net", "org", "inc", "co", "the", "of", "and"}


def _brand_in_domain(brand: str, host: str) -> bool:
    cb = canonical_brand(brand)
    if not cb or not host:
        return False
    toks = [t for t in cb.split() if len(t) >= 4 and t not in _STOP] or [cb]
    flat = host.replace(".", "").replace("-", "")
    return any(t in flat for t in toks)


def consistency_for_brand(brand: str, page: PageRecord) -> str:
    """'consistent' if the (panel-agreed) brand appears in the page's own domain, else
    'inconsistent'. This is the brand-vs-infrastructure check: a brand on its own domain is
    consistent (legit); a brand on an unrelated domain is the phishing signal."""
    host = registrable_domain(page.final_url or page.url) or ""
    return "consistent" if _brand_in_domain(brand, host) else "inconsistent"


def build_consistency_cue(brand: str, page: PageRecord) -> Cue | None:
    """Build the consistency cue for the brand the panel agreed the page presents."""
    if not brand:
        return None
    val = consistency_for_brand(brand, page)
    return Cue(type=CueType.BRAND_DOMAIN_CONSISTENCY, value=val,
               raw_value=brand, asserted_by="tool")


def verify_brand_domain_consistency(cue: Cue, page: PageRecord) -> float:
    """Grounds the consistency cue: re-derive consistent/inconsistent for the cue's brand
    against the page domain and check it matches the claim."""
    brand = cue.raw_value
    if not brand:
        return 0.0
    return 1.0 if cue.value.strip().lower() == consistency_for_brand(brand, page) else 0.0
