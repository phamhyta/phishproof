"""Dispatch a cue to the verifier for its type."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..schema import Cue, CueType, GroundingResult, PageRecord
from . import cert, dom, redirect
from .brand_claim import verify_brand_claim
from .logo import CLIPLogoEmbedder, verify_logo_brand


@dataclass
class Verifiers:
    """Shared resources of the verifiers, set up once per run."""

    logo_embedder: CLIPLogoEmbedder | None = None


def _has_dom(page: PageRecord) -> bool:
    return bool(page.dom_html_path) and Path(page.dom_html_path).exists()


def _score(cue: Cue, page: PageRecord, verifiers: Verifiers) -> tuple[float | None, str]:
    has_dom = _has_dom(page)
    if cue.type is CueType.FORM_ACTION_DOMAIN:
        return (dom.verify_form_action_domain(cue, page) if has_dom else None), "dom.form_action"
    if cue.type is CueType.CREDENTIAL_INTENT:
        return (dom.verify_credential_intent(cue, page) if has_dom else None), "dom.credential"
    if cue.type is CueType.BRAND_CLAIM:
        return (verify_brand_claim(cue, page) if has_dom else None), "html.brand"
    if cue.type is CueType.LOGO_BRAND:
        return verify_logo_brand(cue, page, verifiers.logo_embedder), "clip.logo"
    if cue.type is CueType.CERTIFICATE_ORG:
        return cert.verify_certificate_org(cue, page), "cert.org"
    return redirect.verify_redirect_target(cue, page), "redirect.target"


def verify_cue(
    cue: Cue, page: PageRecord, verifiers: Verifiers | None = None
) -> GroundingResult | None:
    """Re-derive one cue from the capture. None means that the check is unavailable."""
    score, tool = _score(cue, page, verifiers or Verifiers())
    return None if score is None else GroundingResult(cue=cue, score=float(score), tool=tool)
