"""Certificate verifier.

Grounds a certificate/issuer cue when the corpus provides it; inert on captures
that carry no certificate metadata (the cue then simply never grounds).
"""

from __future__ import annotations

from .._release import pending
from ..schema import Cue, GroundingResult
from ..types import PageRecord


class CertificateVerifier:
    """Ground a certificate cue against the page's TLS metadata."""

    name = "certificate"

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult:
        pending("tools.cert.CertificateVerifier.verify")
