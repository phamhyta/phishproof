"""Typed evidence cue schema.

PhishProof reasons over typed, tool-checkable cues rather than free-text
rationales. This module enumerates the cue types and their modality. The schema
is public; the extractors and verifiers that populate it are released with the
paper.
"""

from __future__ import annotations

from enum import Enum


class CueType(str, Enum):
    """The evidence types an agent may cite."""

    BRAND = "brand"                          # impersonated brand identity
    FORM_ACTION = "form_action"              # where a credential form posts
    CREDENTIAL_INTENT = "credential_intent"  # the page solicits credentials
    LOGO = "logo"                            # a rendered brand logo
    CONSISTENCY = "consistency"              # cross-signal coherence


class Modality(str, Enum):
    """Which view of the page a cue comes from."""

    VISION = "vision"        # the rendered screenshot
    STRUCTURE = "structure"  # the page structure / DOM / URL


#: Cue types the per-type agreement score is averaged over (see Section 4).
ACTIVE_CUE_TYPES: tuple[CueType, ...] = (
    CueType.BRAND,
    CueType.FORM_ACTION,
    CueType.CREDENTIAL_INTENT,
    CueType.LOGO,
)

#: Which verifier modality grounds each cue type.
CUE_MODALITY: dict[CueType, Modality] = {
    CueType.BRAND: Modality.STRUCTURE,
    CueType.FORM_ACTION: Modality.STRUCTURE,
    CueType.CREDENTIAL_INTENT: Modality.STRUCTURE,
    CueType.LOGO: Modality.VISION,
    CueType.CONSISTENCY: Modality.STRUCTURE,
}
