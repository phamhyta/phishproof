"""T8 — the frozen COMPLETE verification policy.

This module is the single place that decides act/abstain. It exists because the recorded
evaluation used a SURROGATE gate (nonempty relaxed consensus + mean grounding >= 0.999,
scripts/revision_matched.py) that differs from the policy in Algorithm 1 / Figure 2. The
complete gate here is:

    ACT  iff  valid inference
          AND calibrated trust >= tau
          AND strict-majority consensus is nonempty
          AND every consensus check is AVAILABLE
          AND every consensus check PASSES its frozen threshold
    otherwise ABSTAIN, with an explicit reason.

Frozen rules (policy version below; changing any of them must bump the version):

  * Consensus membership: a cue (type, normalized value) asserted by a STRICT MAJORITY
    (> M/2) of the M panel agents. No relaxation: a single-source perceptual cue and the
    tool-derived brand-domain consistency cue are RECORDED (for the named ablations) but
    are never consensus members here.
  * Validity (required model outputs): every panel agent must have a stored raw response
    (a missing one is a model/API failure) that parses into schema-valid JSON with a
    verdict; the VISION agent must additionally report a numeric confidence in [0,1]
    (it is the trust signal's input). Text-agent confidences are recorded, not required.
    Both captures (screenshot, DOM html) must exist. Any violation is a PERMANENT
    abstention: the page can never be acted on at any threshold.
  * Check availability: a verifier that cannot decide (tool returns N/A, missing CLIP
    dependency, no logo box, missing capture) makes the cue UNAVAILABLE. An unavailable
    consensus check never defaults to passing; it forces abstention.
  * Pass thresholds: structural tools are deterministic 0/1 and pass only at 1.0. The
    perceptual logo check passes at score >= t_logo, where t_logo is selected on
    INDEPENDENT calibration annotations and frozen before test use (revision_t8_policy.py
    writes it into the frozen policy file; the old diagnostic 0.5 is not evidence of a
    calibrated threshold and is not used).

Everything here is pure (no I/O, no model calls) so the unit tests in
tests/test_t8_policy.py can enumerate the acceptance cases exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..schema import PERCEPTUAL_CUE_TYPES, AgentOutput, Cue
from .consensus import _asserting_agents, shared_cue_set
from .normalize import normalize_cue

POLICY_VERSION = "t8-complete-v1"

STRUCTURAL_PASS_THRESHOLD = 1.0  # structural tools return 0/1; only 1.0 passes


class Reason(str, Enum):
    """Canonical abstention reasons, ordered from most to least permanent."""

    MODEL_CALL_MISSING = "invalid:model_call_missing"
    INVALID_JSON = "invalid:invalid_json"
    MISSING_VISION_CONFIDENCE = "invalid:missing_vision_confidence"
    MISSING_CAPTURE = "invalid:missing_capture"
    EMPTY_CONSENSUS = "empty_consensus"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    CHECK_FAILED = "check_failed"
    SCORE_BELOW_TAU = "score_below_tau"
    ACT = "act"


# Order used to pick the PRIMARY reason when several hold. Validity failures are the most
# permanent (no threshold changes them), then evidence-side failures, then the score gate.
_REASON_ORDER = [
    Reason.MODEL_CALL_MISSING,
    Reason.INVALID_JSON,
    Reason.MISSING_VISION_CONFIDENCE,
    Reason.MISSING_CAPTURE,
    Reason.EMPTY_CONSENSUS,
    Reason.VERIFIER_UNAVAILABLE,
    Reason.CHECK_FAILED,
    Reason.SCORE_BELOW_TAU,
]


@dataclass
class AgentValidity:
    """Strict validity of one agent's stored raw response (re-parsed from the cache)."""

    agent_id: str
    model: str
    is_vision: bool
    raw_present: bool          # False = the model call never produced a stored response
    valid_json: bool = False   # schema-valid RawAgentResponse after brace extraction
    has_confidence: bool = False
    n_cues: int = 0

    def failures(self) -> list[Reason]:
        out: list[Reason] = []
        if not self.raw_present:
            out.append(Reason.MODEL_CALL_MISSING)
        elif not self.valid_json:
            out.append(Reason.INVALID_JSON)
        elif self.is_vision and not self.has_confidence:
            out.append(Reason.MISSING_VISION_CONFIDENCE)
        return out


@dataclass
class CueCheck:
    """One candidate cue's full verification record (plan T8: every candidate cue)."""

    type: str                      # CueType value
    value: str                     # normalized value
    raw_values: list[str] = field(default_factory=list)
    asserted_by: list[str] = field(default_factory=list)  # agent ids; [] for derived
    n_asserting: int = 0
    m_agents: int = 0
    source: str = "model"          # "model" (cited by agents) | "derived" (tool-built)
    in_strict_consensus: bool = False
    perceptual: bool = False
    tool: str | None = None
    tool_version: str | None = None
    raw_score: float | None = None
    available: bool = False
    unavailable_reason: str | None = None
    threshold: float = STRUCTURAL_PASS_THRESHOLD
    passed: bool | None = None     # None whenever unavailable — never defaults to passing

    def finalize(self) -> "CueCheck":
        if self.raw_score is None:
            self.available = False
            self.passed = None      # missing groundedness must never pass
        else:
            self.available = True
            self.unavailable_reason = None
            self.passed = self.raw_score >= self.threshold
        return self


@dataclass
class PolicyDecision:
    """The complete-gate outcome for one page, with every intermediate check exposed."""

    valid_inference: bool
    invalid_reasons: list[str]
    n_consensus: int
    consensus_nonempty: bool
    all_available: bool
    all_pass: bool
    trust_calibrated: float | None
    tau: float
    score_pass: bool
    action: str                    # "act" | "abstain"
    primary_reason: str            # Reason value ("act" when acting)
    failed_reasons: list[str]      # every failing condition, canonical order


def build_cue_checks(
    outputs: list[AgentOutput],
    logo_threshold: float,
    derived_cue: Cue | None = None,
) -> list[CueCheck]:
    """Candidate cue records from the panel's cited cues (+ the derived cue, non-member).

    Membership is the STRICT MAJORITY rule only. Grounding scores are attached later by
    the caller (this function is pure); each check is then .finalize()d.
    """
    m = len(outputs)
    pool = shared_cue_set(outputs)
    asserting = _asserting_agents(outputs)
    raw_by_key: dict[tuple[str, str], list[str]] = {}
    for out in outputs:
        for cue in out.cues:
            nc = normalize_cue(cue)
            if nc is None:
                continue
            raw_by_key.setdefault(nc.key(), []).append(cue.value)

    checks: list[CueCheck] = []
    for key, cue in pool.items():
        n = len(asserting[key])
        perceptual = cue.type in PERCEPTUAL_CUE_TYPES
        checks.append(CueCheck(
            type=cue.type.value,
            value=cue.value,
            raw_values=sorted(set(raw_by_key.get(key, []))),
            asserted_by=sorted(asserting[key]),
            n_asserting=n,
            m_agents=m,
            source="model",
            in_strict_consensus=n > m / 2,
            perceptual=perceptual,
            threshold=logo_threshold if perceptual else STRUCTURAL_PASS_THRESHOLD,
        ))
    if derived_cue is not None:
        checks.append(CueCheck(
            type=derived_cue.type.value,
            value=derived_cue.value,
            raw_values=[derived_cue.raw_value or derived_cue.value],
            asserted_by=[],
            n_asserting=0,
            m_agents=m,
            source="derived",
            in_strict_consensus=False,   # named ablation only, never a member here
            perceptual=False,
            threshold=STRUCTURAL_PASS_THRESHOLD,
        ))
    return checks


def evaluate(
    validities: list[AgentValidity],
    capture_ok: dict[str, bool],
    cue_checks: list[CueCheck],
    trust_calibrated: float | None,
    tau: float,
) -> PolicyDecision:
    """Apply the complete gate. Pure; all inputs precomputed by the caller."""
    failed: list[Reason] = []
    invalid: list[str] = []

    for v in validities:
        for r in v.failures():
            failed.append(r)
            invalid.append(f"{r.value}:{v.agent_id}")
    for name, ok in capture_ok.items():
        if not ok:
            failed.append(Reason.MISSING_CAPTURE)
            invalid.append(f"{Reason.MISSING_CAPTURE.value}:{name}")

    valid_inference = not invalid

    consensus = [c for c in cue_checks if c.in_strict_consensus]
    consensus_nonempty = len(consensus) > 0
    all_available = consensus_nonempty and all(c.available for c in consensus)
    # all_pass demands every consensus check both available and passing; an unavailable
    # check can never pass (passed is None, which is not True).
    all_pass = consensus_nonempty and all(c.passed is True for c in consensus)

    score_pass = trust_calibrated is not None and trust_calibrated >= tau

    if not consensus_nonempty:
        failed.append(Reason.EMPTY_CONSENSUS)
    elif not all_available:
        failed.append(Reason.VERIFIER_UNAVAILABLE)
    elif not all_pass:
        failed.append(Reason.CHECK_FAILED)
    if not score_pass:
        failed.append(Reason.SCORE_BELOW_TAU)

    act = valid_inference and score_pass and consensus_nonempty and all_available and all_pass
    ordered = [r for r in _REASON_ORDER if r in failed]
    primary = Reason.ACT if act else ordered[0]

    return PolicyDecision(
        valid_inference=valid_inference,
        invalid_reasons=invalid,
        n_consensus=len(consensus),
        consensus_nonempty=consensus_nonempty,
        all_available=all_available,
        all_pass=all_pass,
        trust_calibrated=trust_calibrated,
        tau=tau,
        score_pass=score_pass,
        action="act" if act else "abstain",
        primary_reason=primary.value,
        failed_reasons=[r.value for r in ordered],
    )
