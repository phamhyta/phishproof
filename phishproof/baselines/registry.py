"""Compute baseline trust scores from panel outputs.

Covers the baselines computable from the shared panel outputs alone:
  B1 single-model conf, B2 verbalized, B4 majority vote, B5 Dawid-Skene, B6 MPG-proxy.
B3 (self-consistency) needs k extra samples per page and is produced by its own runner.

All baselines reuse the same panel so the comparison isolates *where* agreement is
measured (experiments.tex setup).
"""

from __future__ import annotations

from ..schema import AgentOutput
from .dawid_skene import dawid_skene
from .label_scores import b1_single_model_confidence, b2_verbalized_confidence, b4_majority_vote
from .multiphishguard import b6_multiphishguard_proxy

# Baselines this module can produce from panel outputs only.
PANEL_BASELINES = ("B1", "B2", "B4", "B5", "B6")


def compute_panel_baselines(
    outputs_by_page: dict[str, list[AgentOutput]],
    primary_agent_id: str | None = None,
) -> dict[str, dict[str, float]]:
    """Return {baseline_id: {page_id: trust_score}} for the panel-derived baselines."""
    per_page = {
        "B1": {}, "B2": {}, "B4": {}, "B6": {},
    }
    for pid, outs in outputs_by_page.items():
        per_page["B1"][pid] = b1_single_model_confidence(outs, primary_agent_id)
        per_page["B2"][pid] = b2_verbalized_confidence(outs)
        per_page["B4"][pid] = b4_majority_vote(outs)
        per_page["B6"][pid] = b6_multiphishguard_proxy(outs)
    per_page["B5"] = dawid_skene(outputs_by_page)  # dataset-level EM
    return per_page
