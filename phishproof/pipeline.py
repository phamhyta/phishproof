"""The end-to-end detector, composed from the panel, verification and policy layers."""

from __future__ import annotations

from .evidence import grounded_evidence_agreement
from .panel import Panel, majority_verdict
from .policy import (
    DEFAULT_LOGO_THRESHOLD,
    Decision,
    FrozenPolicy,
    check_consensus,
    decide,
    groundedness,
    mean_confidence,
)
from .schema import MemberOutput, PageEvidence, PageRecord
from .verification import BrandDomains, Verifiers, binding_holds, named_brands


def build_evidence(
    page: PageRecord,
    outputs: list[MemberOutput],
    logo_threshold: float = DEFAULT_LOGO_THRESHOLD,
    brand_domains: BrandDomains | None = None,
    verifiers: Verifiers | None = None,
) -> PageEvidence:
    """Assemble what the policy reads from the panel's outputs and the page capture."""
    brands = sorted(named_brands(outputs))
    checks = check_consensus(outputs, page, logo_threshold, verifiers)
    return PageEvidence(
        page_id=page.page_id,
        url=page.host_url,
        label=page.label,
        outputs=outputs,
        verdict=majority_verdict(outputs),
        rank=mean_confidence(outputs),
        complete=all(output.confidence is not None for output in outputs),
        gea=grounded_evidence_agreement(outputs),
        groundedness=groundedness(checks),
        checks=checks,
        named_brands=brands,
        bound=binding_holds(brands, page.host_url, brand_domains),
    )


class PhishProof:
    def __init__(
        self,
        panel: Panel,
        policy: FrozenPolicy | None = None,
        brand_domains: BrandDomains | None = None,
        verifiers: Verifiers | None = None,
    ) -> None:
        self.panel = panel
        self.policy = policy
        self.brand_domains = brand_domains
        self.verifiers = verifiers or Verifiers()

    def collect(self, page: PageRecord) -> PageEvidence:
        logo_threshold = self.policy.logo_threshold if self.policy else DEFAULT_LOGO_THRESHOLD
        outputs = self.panel.run(page)
        return build_evidence(page, outputs, logo_threshold, self.brand_domains, self.verifiers)

    def assess(self, page: PageRecord) -> Decision:
        if self.policy is None:
            raise ValueError("assess() needs a frozen policy; see phishproof.fit_policy")
        return decide(self.collect(page), self.policy)
