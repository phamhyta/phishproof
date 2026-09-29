"""Consensus + evidence agreement (C2, eq:agree).

Shared cue set E = union of normalized cues across agents (distinct by (type, value)).
Consensus cons = cues a strict majority of agents assert.
Agreement A = |cons| / |E|  -- high only when agents converge on the SAME evidence.
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import PERCEPTUAL_CUE_TYPES, AgentOutput, Cue, CueType
from .normalize import normalize_cue

# Cue types decidable on the evaluation corpus (cert/redirect are inert here). Per-type
# agreement is averaged over this fixed set, so an absent type contributes 0.
ACTIVE_CUE_TYPES: tuple[CueType, ...] = (
    CueType.BRAND_CLAIM,
    CueType.FORM_ACTION_DOMAIN,
    CueType.CREDENTIAL_INTENT,
    CueType.LOGO_BRAND,
)


def shared_cue_set(outputs: list[AgentOutput]) -> dict[tuple[str, str], Cue]:
    """Union of normalized cues, keyed by (type, value). One representative Cue each."""
    pool: dict[tuple[str, str], Cue] = {}
    for out in outputs:
        for cue in out.cues:
            nc = normalize_cue(cue)
            if nc is None:
                continue
            pool.setdefault(nc.key(), nc)
    return pool


def _asserting_agents(outputs: list[AgentOutput]) -> dict[tuple[str, str], set[str]]:
    agents: dict[tuple[str, str], set[str]] = defaultdict(set)
    for out in outputs:
        for cue in out.cues:
            nc = normalize_cue(cue)
            if nc is not None:
                agents[nc.key()].add(out.agent_id)
    return agents


def consensus_cues(outputs: list[AgentOutput], relax_perceptual: bool = False) -> list[Cue]:
    """Cues asserted by a strict majority (> M/2) of the M agents.

    With relax_perceptual=True, a single-source PERCEPTUAL cue (e.g. logo_brand, which only
    the one VLM can see) may enter the consensus on its own — it cannot be corroborated by
    text agents that never look at the screenshot, so requiring a majority would always
    exclude it.
    """
    m = len(outputs)
    pool = shared_cue_set(outputs)
    counts = _asserting_agents(outputs)
    out = []
    for key, cue in pool.items():
        n = len(counts[key])
        if n > m / 2:
            out.append(cue)
        elif relax_perceptual and cue.type in PERCEPTUAL_CUE_TYPES and n >= 1:
            out.append(cue)
    return out


def agreement(outputs: list[AgentOutput], relax_perceptual: bool = False) -> tuple[float, list[Cue]]:
    """Return (A, consensus cues). A = |consensus| / |shared cue set|; 0 if no cues."""
    pool = shared_cue_set(outputs)
    if not pool:
        return 0.0, []
    cons = consensus_cues(outputs, relax_perceptual)
    return len(cons) / len(pool), cons


def per_type_agreement(
    outputs: list[AgentOutput], active_types: tuple[CueType, ...] = ACTIVE_CUE_TYPES
) -> tuple[float, dict[CueType, float]]:
    """Parameter-free evidence agreement (eq:agree-pt): mean per-type agreement.

    For each active cue type t, a_t = (max number of agents asserting any single normalized
    value for t) / M -- the strength of the panel's agreement on that evidence type; an absent
    type contributes 0. GEA = mean_t a_t over the fixed active set: a training-free combination
    with no learned weights. Unlike the consensus fraction |cons|/|E|, this is not diluted by
    cue count and weights each evidence type equally, which gives a finer-grained, less
    saturated trust signal (the high-end ranking is what selective prediction needs).
    """
    m = len(outputs)
    if m == 0 or not active_types:
        return 0.0, {}
    per_type: dict[CueType, float] = {}
    for t in active_types:
        by_val: dict[str, set[str]] = defaultdict(set)
        for out in outputs:
            for cue in out.cues:
                if cue.type is t:
                    nc = normalize_cue(cue)
                    if nc is not None:
                        by_val[nc.value].add(out.agent_id)
        per_type[t] = max((len(s) for s in by_val.values()), default=0) / m
    return sum(per_type.values()) / len(active_types), per_type


def router_gates(
    outputs: list[AgentOutput], ground_score: dict[tuple[str, str], float | None]
) -> dict[str, float]:
    """LEGACY -- NOT used by the headline method. Only the legacy ``score_mode="product"`` path
    (gea.py) calls this; the deployed detector ranks by the calibrated trust ``s`` and forms
    consensus/veto from per-type agreement, with no router and no trained gate (see related.tex).
    Kept solely for the two-tier / routing ablations; do not read it as part of the main method.

    Grounded router gate r_i per expert (eq:gate).

    r_i = mean over agent i's OWN normalized cued cues of the tool grounding score g(u,x) in
    [0,1]; N/A cues (no decisive tool, e.g. cert/redirect on this corpus) are excluded. r_i=0
    when the agent cites no groundable cue. ground_score maps a cue key to its grounding,
    computed once over the shared cue set. The gates are aggregated into the panel routing
    confidence R that scales the trust score (gea.py), NOT into the consensus membership, so
    the count-based agreement A is unchanged and the router only re-weights the trust.
    """
    gates: dict[str, float] = {}
    for out in outputs:
        scores: list[float] = []
        seen: set[tuple[str, str]] = set()
        for cue in out.cues:
            nc = normalize_cue(cue)
            if nc is None or nc.key() in seen:
                continue
            seen.add(nc.key())
            g = ground_score.get(nc.key())
            if g is not None:
                scores.append(g)
        gates[out.agent_id] = sum(scores) / len(scores) if scores else 0.0
    return gates


def routing_confidence(gates: dict[str, float], theta: float = 0.7) -> float:
    """LEGACY -- NOT used by the headline method (see ``router_gates`` above; legacy
    ``score_mode="product"`` only).

    Panel routing confidence R in [0,1].

    R = min(1, mean_gate / theta) over the ACTIVE experts (those that cited at least one
    groundable cue, i.e. gate > 0); R=1 when no expert is active. The clip at theta makes the
    router \"route in at full confidence\" any panel whose experts ground above theta -- so on
    clean pages, where experts ground well, R=1 and the score is the plain A*G (the clean
    ranking is preserved). When an expert's cited evidence is corrupted its gate falls; once
    the active-expert mean drops below theta, R<1, GEA=A*G*R collapses, and the detector
    abstains. theta is a hyperparameter (swept like the threshold tau and the consensus k)."""
    active = [v for v in gates.values() if v > 0]
    if not active:
        return 1.0
    return min(1.0, (sum(active) / len(active)) / theta)
