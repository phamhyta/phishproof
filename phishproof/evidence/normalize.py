"""Canonical cue values, so that two members can assert the same cue in different words."""

from __future__ import annotations

from ..schema import Cue, CueType
from ..verification.brands import canonical_brand
from ..verification.urls import registrable_domain

_DOMAIN_TYPES = {CueType.FORM_ACTION_DOMAIN, CueType.REDIRECT_TARGET}
_BRAND_TYPES = {CueType.BRAND_CLAIM, CueType.LOGO_BRAND}


def normalize_value(cue_type: CueType, value: str) -> str | None:
    if cue_type in _DOMAIN_TYPES:
        return registrable_domain(value)
    if cue_type in _BRAND_TYPES:
        return canonical_brand(value)
    if cue_type is CueType.CREDENTIAL_INTENT:
        return "yes" if value.strip().lower() in ("yes", "true", "1", "present", "phish") else "no"
    return value.strip().lower() or None


def normalize_cue(cue: Cue) -> Cue | None:
    value = normalize_value(cue.type, cue.value)
    if not value:
        return None
    return cue.model_copy(update={"value": value, "raw_value": cue.raw_value or cue.value})
