"""The evidence-cue schema: typed claims that a verifier can re-derive from a capture."""

from __future__ import annotations

from enum import Enum


class CueType(str, Enum):
    BRAND_CLAIM = "brand_claim"
    FORM_ACTION_DOMAIN = "form_action_domain"
    CERTIFICATE_ORG = "certificate_org"
    REDIRECT_TARGET = "redirect_target"
    CREDENTIAL_INTENT = "credential_intent"
    LOGO_BRAND = "logo_brand"


# Archived captures have no certificate or redirect chain, so those two types stay inactive.
ACTIVE_CUE_TYPES: tuple[CueType, ...] = (
    CueType.BRAND_CLAIM,
    CueType.FORM_ACTION_DOMAIN,
    CueType.CREDENTIAL_INTENT,
    CueType.LOGO_BRAND,
)

# Only these two checks may gate an action. Brand and logo checks are kept for the audit trail.
GATING_CUE_TYPES: frozenset[CueType] = frozenset(
    {CueType.FORM_ACTION_DOMAIN, CueType.CREDENTIAL_INTENT}
)
BRAND_CUE_TYPES: frozenset[CueType] = frozenset({CueType.BRAND_CLAIM, CueType.LOGO_BRAND})
PERCEPTUAL_CUE_TYPES: frozenset[CueType] = frozenset({CueType.LOGO_BRAND})
