"""Label-level / confidence baselines computable directly from panel outputs.

  B1 single-model confidence (guo2017calibration): one model's confidence.
  B2 verbalized confidence:                        the model's stated confidence.
  B4 majority vote (ensembleornot2024):            panel label-agreement share.

B1/B2 use the agents' confidence field (self-reported). With local Ollama models that is
a verbalized number; a temperature-scaled softmax variant can replace it once logprobs
are available. B3 (self-consistency) and B6 (MultiPhishGuard) need extra sampling/agents
and live in their own runners.
"""

from __future__ import annotations

from collections import Counter

from ..schema import AgentOutput


def b4_majority_vote(outputs: list[AgentOutput]) -> float:
    """Share of agents agreeing with the majority label (label concordance)."""
    if not outputs:
        return 0.0
    counts = Counter(o.verdict for o in outputs)
    return counts.most_common(1)[0][1] / len(outputs)


def b1_single_model_confidence(outputs: list[AgentOutput], primary_agent_id: str | None = None) -> float:
    """Confidence of the designated strongest single model (default: first agent)."""
    if not outputs:
        return 0.0
    chosen = outputs[0]
    if primary_agent_id:
        for o in outputs:
            if o.agent_id == primary_agent_id:
                chosen = o
                break
    return float(chosen.confidence) if chosen.confidence is not None else 1.0


def b2_verbalized_confidence(outputs: list[AgentOutput]) -> float:
    """Mean self-reported (verbalized) confidence across the panel."""
    vals = [o.confidence for o in outputs if o.confidence is not None]
    return float(sum(vals) / len(vals)) if vals else 1.0
