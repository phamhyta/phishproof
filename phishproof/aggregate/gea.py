"""Grounded Evidence-Agreement (GEA) scoring."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..config import ScoreConfig
from ..schema import AgentOutput, GroundingResult, TrustScore


def gea_score(outputs: Sequence[AgentOutput], config: ScoreConfig) -> float:
    """The parameter-free GEA: the uniform mean of per-type agreement.

    There are no learned weights; the predictive signal lives in the grounded
    evidence, not in the combiner.
    """
    pending("aggregate.gea.gea_score")


def trust_signal(
    outputs: Sequence[AgentOutput],
    vision_confidence: float,
    config: ScoreConfig,
) -> float:
    """Deployed ranking signal ``s = conf_VLM + epsilon * mean_text_agreement``.

    The vision confidence is primary; the text agreement breaks ties among the
    discrete confidence levels.
    """
    pending("aggregate.gea.trust_signal")


def score_page(
    outputs: Sequence[AgentOutput],
    grounding: Sequence[GroundingResult],
    config: ScoreConfig,
    *,
    score_mode: str = "agreement",
) -> TrustScore:
    """Combine agreement, the grounding veto, and calibration into a TrustScore.

    ``score_mode='agreement'`` uses the parameter-free per-type mean (default);
    ``'product'`` uses the grounded-product variant that carries the Prop-1
    guarantee. A failed grounding check vetoes the action regardless of how
    strongly the panel agrees.
    """
    pending("aggregate.gea.score_page")
