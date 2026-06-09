"""Value normalization (C2, method.tex §sec:agree).

Two agents must be able to assert the "same" cue despite surface differences, so values
are canonicalized by type before consensus: domains -> eTLD+1, brands -> canonical
lexicon, credential-intent -> yes/no.
"""

from __future__ import annotations

from ..schema import Cue, CueType
from ..tools.brands import canonical_brand
from ..tools.urls import registrable_domain

_DOMAIN_TYPES = {CueType.FORM_ACTION_DOMAIN, CueType.REDIRECT_TARGET}
_BRAND_TYPES = {CueType.BRAND_CLAIM, CueType.LOGO_BRAND}


def _norm_bool(value: str) -> str:
    v = value.strip().lower()
    return "yes" if v in ("yes", "true", "1", "present", "phish") else "no"


def normalize_value(ctype: CueType, value: str) -> str | None:
    if ctype in _DOMAIN_TYPES:
        return registrable_domain(value)
    if ctype in _BRAND_TYPES:
        return canonical_brand(value)
    if ctype is CueType.CREDENTIAL_INTENT:
        return _norm_bool(value)
    if ctype is CueType.BRAND_DOMAIN_CONSISTENCY:
        v = value.strip().lower()
        return "consistent" if v.startswith("cons") else "inconsistent"
    if ctype is CueType.CERTIFICATE_ORG:
        return value.strip().lower() or None
    return value.strip() or None


def normalize_cue(cue: Cue) -> Cue | None:
    """Return a copy with a normalized value, or None if the value normalizes to nothing."""
    nv = normalize_value(cue.type, cue.value)
    if not nv:
        return None
    return cue.model_copy(update={"value": nv, "raw_value": cue.raw_value or cue.value})
