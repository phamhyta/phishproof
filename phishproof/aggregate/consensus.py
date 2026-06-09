"""Consensus + evidence agreement (C2, eq:agree).

Shared cue set E = union of normalized cues across agents (distinct by (type, value)).
Consensus cons = cues a strict majority of agents assert.
Agreement A = |cons| / |E|  -- high only when agents converge on the SAME evidence.
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import PERCEPTUAL_CUE_TYPES, AgentOutput, Cue
from .normalize import normalize_cue


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
