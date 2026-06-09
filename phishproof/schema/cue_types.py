"""The fixed evidence-cue schema U (method.tex §sec:elicit).

Six typed cue types, each a tool-checkable (type, value) claim. Agents emit cues
*only* from this set via constrained decoding, so cues are directly comparable and
auditable. Each cue type has a tool that can re-derive it (grounding, §sec:ground).

Per decision 0b (Phishpedia 30k has no TLS cert / redirect chain), only four cue
types are ACTIVE for the current run; the other two stay defined so they can be
re-enabled by a backfill capture without touching the schema.
"""

from __future__ import annotations

from enum import Enum


class CueType(str, Enum):
    BRAND_CLAIM = "brand_claim"              # brand the page poses as (canonical)
    FORM_ACTION_DOMAIN = "form_action_domain"  # eTLD+1 a form posts to
    CERTIFICATE_ORG = "certificate_org"      # organization on the TLS certificate
    REDIRECT_TARGET = "redirect_target"      # eTLD+1 final target of the redirect chain
    CREDENTIAL_INTENT = "credential_intent"  # whether the page collects credentials
    LOGO_BRAND = "logo_brand"                # brand of the rendered logo
    BRAND_DOMAIN_CONSISTENCY = "brand_domain_consistency"  # brand matches page domain?


# --- Grounding semantics: how each cue type's tool returns a score in [0,1] ---
# Structural tools verify deterministically and return 0 or 1.
STRUCTURAL_CUE_TYPES: frozenset[CueType] = frozenset({
    CueType.FORM_ACTION_DOMAIN,
    CueType.CERTIFICATE_ORG,
    CueType.REDIRECT_TARGET,
    CueType.CREDENTIAL_INTENT,
    CueType.BRAND_DOMAIN_CONSISTENCY,
})
# Perceptual tool returns a similarity in [0,1].
PERCEPTUAL_CUE_TYPES: frozenset[CueType] = frozenset({CueType.LOGO_BRAND})
# Brand claim is grounded by a specialized detector used as a tool (D1/D2/D3).
DETECTOR_CUE_TYPES: frozenset[CueType] = frozenset({CueType.BRAND_CLAIM})

# --- Active set for the current Phishpedia-based run (decision 0b) ---
ACTIVE_CUE_TYPES: tuple[CueType, ...] = (
    CueType.BRAND_CLAIM,
    CueType.FORM_ACTION_DOMAIN,
    CueType.CREDENTIAL_INTENT,
    CueType.LOGO_BRAND,
)
# Defined but inactive until cert/redirect are backfilled.
INACTIVE_CUE_TYPES: tuple[CueType, ...] = (
    CueType.CERTIFICATE_ORG,
    CueType.REDIRECT_TARGET,
)
