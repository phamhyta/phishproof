"""End-to-end GEA scoring for one page (Algorithm 1, minus calibration -> Phase 4).

  A   = evidence agreement over the shared cue set        (consensus.py)
  G   = mean tool score over consensus, dropping N/A cues (tools.registry)
  GEA = A * G                                              (eq:gea)

By Proposition 1, a consensus cue that does not hold contributes 0 to G, so a fully
hallucinated consensus drives GEA to 0.
"""

from __future__ import annotations

from ..schema import AgentOutput, Cue, CueType, GEAResult, PageRecord
from ..tools.consistency import build_consistency_cue
from ..tools.registry import GroundingContext, ground_cue
from .consensus import agreement, shared_cue_set


def groundedness(cons: list[Cue], page: PageRecord, ctx: GroundingContext | None = None):
    """G = mean tool score over consensus cues whose tool can decide (N/A dropped).

    Returns (G, grounded_results, n_na). G is 0.0 if every consensus cue is N/A.
    """
    results = []
    for cue in cons:
        r = ground_cue(cue, page, ctx)
        if r is not None:
            results.append(r)
    n_na = len(cons) - len(results)
    g = sum(r.score for r in results) / len(results) if results else 0.0
    return g, results, n_na


def score_page(
    outputs: list[AgentOutput],
    page: PageRecord,
    ctx: GroundingContext | None = None,
    add_consistency: bool = False,
    relax_perceptual: bool = False,
) -> GEAResult:
    from ..agents.panel import Panel

    a, cons = agreement(outputs, relax_perceptual=relax_perceptual)

    # Add the brand-domain consistency cue derived from the brand the panel AGREED the page
    # presents (the consensus brand_claim) checked against the page's own domain. Symmetric
    # for phish/benign and uses no gold label: a legit brand on its own domain grounds
    # 'consistent' (giving benign pages positive grounded evidence), an impersonated brand on
    # an unrelated domain grounds 'inconsistent'. Only added when the panel agreed on a brand.
    if add_consistency:
        brand = next((c.value for c in cons if c.type is CueType.BRAND_CLAIM), None)
        ccue = build_consistency_cue(brand, page) if brand else None
        if ccue is not None:
            n_shared = len(shared_cue_set(outputs)) + 1
            cons = list(cons) + [ccue]
            a = len(cons) / n_shared if n_shared else 0.0

    g, _results, _na = groundedness(cons, page, ctx)
    gea = a * g
    return GEAResult(
        page_id=page.page_id,
        verdict=Panel.majority_label(outputs),
        agreement=a,
        groundedness=g,
        gea=gea,
        consensus_cues=cons,
    )
