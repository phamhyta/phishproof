"""End-to-end GEA scoring for one page (Algorithm 1, minus calibration -> Phase 4).

Default (method D, score_mode="agreement") -- the headline method (eq:gea):
  GEA = mean_t a_t  over the active evidence types,
  a_t = (max agents citing a single value for type t) / M           (consensus.per_type_agreement)
a parameter-free uniform average, no trained parameters. Grounding G (mean tool score over the
consensus, N/A cues dropped) is still computed and stored -- for the grounding veto / adversarial
fail-safe (RQ7) and the audit trail returned with each action -- but it does NOT enter the
clean-data ranking: on clean pages it is near-constant, consistent with the grounding-neutral
ablation (RQ3).

Legacy (score_mode="product") -- NOT the headline method, kept for the two-tier comparison and
the routing/grounding ablations: the zero-parameter grounded product GEA = A * G * R, where
A = consensus evidence agreement, G as above, and R = panel routing confidence (mean grounded
gate of the active experts; routing=False sets R=1, recovering plain A*G). By Proposition 1 a
fully hallucinated consensus drives the product to 0.
"""

from __future__ import annotations

from ..schema import AgentOutput, Cue, CueType, GEAResult, GroundingResult, PageRecord
from ..tools.consistency import build_consistency_cue
from ..tools.registry import GroundingContext, ground_cue
from .consensus import (
    agreement,
    per_type_agreement,
    router_gates,
    routing_confidence,
    shared_cue_set,
)


def _ground_shared_set(
    pool: dict[tuple[str, str], Cue], page: PageRecord, ctx: GroundingContext | None
) -> tuple[dict[tuple[str, str], GroundingResult | None], dict[tuple[str, str], float | None]]:
    """Ground every cited cue in the shared set once -> (results-by-key, score-by-key).

    The single pass feeds both the router gates (per-expert) and the consensus groundedness G,
    so no cue is grounded twice.
    """
    res: dict[tuple[str, str], GroundingResult | None] = {}
    score: dict[tuple[str, str], float | None] = {}
    for key, cue in pool.items():
        r = ground_cue(cue, page, ctx)
        res[key] = r
        score[key] = r.score if r is not None else None
    return res, score


def _consensus_groundedness(
    cons: list[Cue],
    ground_res: dict[tuple[str, str], GroundingResult | None],
    page: PageRecord,
    ctx: GroundingContext | None,
) -> float:
    """G = mean tool score over consensus cues whose tool can decide (N/A dropped, eq:ground).

    Reuses the single grounding pass; the derived brand-domain consistency cue is not in the
    shared set, so it is grounded once here.
    """
    results: list[GroundingResult] = []
    for cue in cons:
        key = cue.key()
        r = ground_res[key] if key in ground_res else ground_cue(cue, page, ctx)
        if r is not None:
            results.append(r)
    return sum(r.score for r in results) / len(results) if results else 0.0


# Back-compat shim: some scripts import groundedness() directly.
def groundedness(cons: list[Cue], page: PageRecord, ctx: GroundingContext | None = None):
    results = [r for cue in cons if (r := ground_cue(cue, page, ctx)) is not None]
    n_na = len(cons) - len(results)
    g = sum(r.score for r in results) / len(results) if results else 0.0
    return g, results, n_na


def score_page(
    outputs: list[AgentOutput],
    page: PageRecord,
    ctx: GroundingContext | None = None,
    add_consistency: bool = False,
    relax_perceptual: bool = False,
    routing: bool = True,
    score_mode: str = "agreement",
) -> GEAResult:
    from ..agents.panel import Panel

    pool = shared_cue_set(outputs)
    # ground every cited cue once -> feeds router gates AND consensus groundedness
    ground_res, ground_score = _ground_shared_set(pool, page, ctx)

    # count-based strict-majority consensus + evidence agreement (unchanged by the router)
    a, cons = agreement(outputs, relax_perceptual=relax_perceptual)

    # Derived brand-domain consistency cue from the AGREED brand checked against the domain.
    if add_consistency:
        brand = next((c.value for c in cons if c.type is CueType.BRAND_CLAIM), None)
        ccue = build_consistency_cue(brand, page) if brand else None
        if ccue is not None:
            n_shared = len(pool) + 1
            cons = list(cons) + [ccue]
            a = len(cons) / n_shared if n_shared else 0.0

    g = _consensus_groundedness(cons, ground_res, page, ctx)

    # soft grounded router: R re-weights the trust (R=1 with routing off)
    r = routing_confidence(router_gates(outputs, ground_score)) if routing else 1.0

    # Trust score. "agreement" (default, method D): parameter-free mean per-type agreement --
    # grounding (g) verifies the cited evidence and provides the adversarial fail-safe but does
    # not enter the clean-data ranking (it is near-constant on clean pages). "product": the
    # legacy zero-param grounded product A*G*R (Prop 1 holds: a fully hallucinated consensus
    # drives gea to 0), kept for the two-tier comparison and the routing/grounding ablations.
    pta, _ = per_type_agreement(outputs)
    gea = pta if score_mode == "agreement" else a * g * r
    return GEAResult(
        page_id=page.page_id,
        verdict=Panel.majority_label(outputs),
        agreement=a,
        groundedness=g,
        gea=gea,
        consensus_cues=cons,
    )
