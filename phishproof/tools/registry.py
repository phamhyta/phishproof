"""Grounding dispatch: cue -> the tool for its type -> GroundingResult (eq:ground).

A score of None means N/A (the cue's tool cannot decide on this page, e.g. cert/redirect
on Phishpedia); the caller drops N/A cues from the consensus before averaging G.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import Cue, CueType, GroundingResult, PageRecord
from . import cert, dom, logo_brand, redirect
from .consistency import verify_brand_domain_consistency
from .detector import BrandDetector, verify_brand_claim
from .logo_brand import CLIPLogoEmbedder


@dataclass
class GroundingContext:
    """Pluggable tools the grounders need (set up once per run)."""

    detector: BrandDetector | None = None        # brand_claim grounding (D1/D2/D3)
    logo_embedder: CLIPLogoEmbedder | None = None  # logo_brand grounding (CLIP)


def ground_cue(
    cue: Cue, page: PageRecord, ctx: GroundingContext | None = None
) -> GroundingResult | None:
    """Re-derive one cue with its tool. Returns None if the tool yields N/A."""
    ctx = ctx or GroundingContext()
    t = cue.type
    score: float | None
    tool: str
    if t is CueType.FORM_ACTION_DOMAIN:
        score, tool = dom.verify_form_action_domain(cue, page), "dom.form_action"
    elif t is CueType.CREDENTIAL_INTENT:
        score, tool = dom.verify_credential_intent(cue, page), "dom.credential_intent"
    elif t is CueType.CERTIFICATE_ORG:
        score, tool = cert.verify_certificate_org(cue, page), "cert.org"
    elif t is CueType.REDIRECT_TARGET:
        score, tool = redirect.verify_redirect_target(cue, page), "redirect.target"
    elif t is CueType.LOGO_BRAND:
        score, tool = logo_brand.verify_logo_brand(cue, page, ctx.logo_embedder), "logo.clip"
    elif t is CueType.BRAND_CLAIM:
        score, tool = verify_brand_claim(cue, page, ctx.detector), "detector.brand"
    elif t is CueType.BRAND_DOMAIN_CONSISTENCY:
        score, tool = verify_brand_domain_consistency(cue, page), "consistency.brand_domain"
    else:  # pragma: no cover - exhaustive over CueType
        return None

    if score is None:
        return None
    return GroundingResult(cue=cue, score=float(score), tool=tool)
