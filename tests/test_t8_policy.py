"""T8 acceptance cases for the complete verification policy.

Every case the plan enumerates has an explicit expected outcome:
  empty consensus, single-source logo, derived-only evidence, missing CLIP dependency,
  one failed cue, one unavailable cue, malformed JSON, missing vision confidence,
  and the clean all-pass case that acts only when the calibrated confidence passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.policy import (
    AgentValidity,
    CueCheck,
    Reason,
    build_cue_checks,
    evaluate,
)
from phishproof.schema import AgentOutput, Cue, CueType, Label

TAU = 0.9651
T_LOGO = 0.62  # any frozen value works for the unit cases; the real one is fit on calib

VISION = "agent_c_vision"


def _valid_panel() -> list[AgentValidity]:
    return [
        AgentValidity("agent_a_text", "llama-3.3-70b", False, True, True, True, 2),
        AgentValidity("agent_b_text", "qwen-2.5-72b", False, True, True, True, 2),
        AgentValidity(VISION, "gpt-4o", True, True, True, True, 2),
    ]


def _captures_ok() -> dict[str, bool]:
    return {"screenshot": True, "html": True}


def _out(agent_id: str, cues: list[Cue]) -> AgentOutput:
    return AgentOutput(agent_id=agent_id, verdict=Label.PHISH, cues=cues,
                       confidence=0.95)


def _brand(v: str = "paypal") -> Cue:
    return Cue(type=CueType.BRAND_CLAIM, value=v)


def _logo(v: str = "paypal") -> Cue:
    return Cue(type=CueType.LOGO_BRAND, value=v)


def _ground(checks: list[CueCheck], scores: dict[tuple[str, str], float | None],
            reasons: dict[tuple[str, str], str] | None = None) -> list[CueCheck]:
    reasons = reasons or {}
    for c in checks:
        key = (c.type, c.value)
        c.raw_score = scores.get(key)
        if c.raw_score is None:
            c.unavailable_reason = reasons.get(key, "tool_na")
        c.finalize()
    return checks


# ---------------------------------------------------------------- acceptance cases

def test_empty_consensus_abstains():
    """No cue reaches a strict majority -> abstain with empty_consensus, even at trust 1.0."""
    outs = [_out("agent_a_text", [_brand("paypal")]),
            _out("agent_b_text", [_brand("apple")]),
            _out(VISION, [_brand("google")])]
    checks = _ground(build_cue_checks(outs, T_LOGO),
                     {("brand_claim", "paypal"): 1.0, ("brand_claim", "apple"): 1.0,
                      ("brand_claim", "google"): 1.0})
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.EMPTY_CONSENSUS.value
    assert d.n_consensus == 0


def test_single_source_logo_is_not_consensus():
    """A logo cue only the VLM cites must NOT enter the strict consensus (no relaxation)."""
    outs = [_out("agent_a_text", []), _out("agent_b_text", []), _out(VISION, [_logo()])]
    checks = build_cue_checks(outs, T_LOGO)
    logo = [c for c in checks if c.type == "logo_brand"][0]
    assert logo.n_asserting == 1
    assert logo.in_strict_consensus is False
    checks = _ground(checks, {("logo_brand", "paypal"): 0.99})
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"                      # nothing else is in consensus
    assert d.primary_reason == Reason.EMPTY_CONSENSUS.value


def test_derived_only_evidence_abstains():
    """Only the tool-derived consistency cue exists -> it is never a member -> abstain."""
    outs = [_out("agent_a_text", []), _out("agent_b_text", []), _out(VISION, [])]
    derived = Cue(type=CueType.BRAND_DOMAIN_CONSISTENCY, value="inconsistent",
                  raw_value="paypal", asserted_by="tool")
    checks = _ground(build_cue_checks(outs, T_LOGO, derived_cue=derived),
                     {("brand_domain_consistency", "inconsistent"): 1.0})
    derived_check = [c for c in checks if c.source == "derived"][0]
    assert derived_check.in_strict_consensus is False
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.EMPTY_CONSENSUS.value


def test_missing_clip_dependency_forces_abstain():
    """A majority logo cue whose CLIP verifier is unavailable can never default to passing."""
    outs = [_out("agent_a_text", [_logo()]), _out("agent_b_text", [_logo()]),
            _out(VISION, [_logo()])]
    checks = _ground(build_cue_checks(outs, T_LOGO),
                     {("logo_brand", "paypal"): None},
                     {("logo_brand", "paypal"): "clip_unavailable"})
    logo = checks[0]
    assert logo.in_strict_consensus and not logo.available and logo.passed is None
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.VERIFIER_UNAVAILABLE.value


def test_one_failed_cue_vetoes():
    """Every consensus check must pass: one failing structural check -> abstain."""
    outs = [_out("agent_a_text", [_brand(), Cue(type=CueType.FORM_ACTION_DOMAIN, value="evil.com")]),
            _out("agent_b_text", [_brand(), Cue(type=CueType.FORM_ACTION_DOMAIN, value="evil.com")]),
            _out(VISION, [_brand(), Cue(type=CueType.FORM_ACTION_DOMAIN, value="evil.com")])]
    checks = _ground(build_cue_checks(outs, T_LOGO),
                     {("brand_claim", "paypal"): 1.0,
                      ("form_action_domain", "evil.com"): 0.0})
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.CHECK_FAILED.value
    assert d.all_available is True and d.all_pass is False


def test_one_unavailable_cue_forces_abstain():
    """One consensus check available+passing, one unavailable -> abstain (no mean tricks)."""
    outs = [_out("agent_a_text", [_brand(), _logo()]),
            _out("agent_b_text", [_brand(), _logo()]),
            _out(VISION, [_brand(), _logo()])]
    checks = _ground(build_cue_checks(outs, T_LOGO),
                     {("brand_claim", "paypal"): 1.0, ("logo_brand", "paypal"): None},
                     {("logo_brand", "paypal"): "no_logo_box"})
    d = evaluate(_valid_panel(), _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.VERIFIER_UNAVAILABLE.value


def test_malformed_json_is_permanent_abstention():
    """An agent whose stored raw response is not schema-valid JSON invalidates inference."""
    vals = _valid_panel()
    vals[0] = AgentValidity("agent_a_text", "llama-3.3-70b", False,
                            raw_present=True, valid_json=False)
    outs = [_out("agent_a_text", [_brand()]), _out("agent_b_text", [_brand()]),
            _out(VISION, [_brand()])]
    checks = _ground(build_cue_checks(outs, T_LOGO), {("brand_claim", "paypal"): 1.0})
    d = evaluate(vals, _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.valid_inference is False
    assert d.primary_reason == Reason.INVALID_JSON.value
    assert "invalid:invalid_json:agent_a_text" in d.invalid_reasons


def test_missing_vision_confidence_is_permanent_abstention():
    """The vision confidence is a required output; without it the page cannot be acted on."""
    vals = _valid_panel()
    vals[2] = AgentValidity(VISION, "gpt-4o", True, raw_present=True, valid_json=True,
                            has_confidence=False)
    outs = [_out("agent_a_text", [_brand()]), _out("agent_b_text", [_brand()]),
            _out(VISION, [_brand()])]
    checks = _ground(build_cue_checks(outs, T_LOGO), {("brand_claim", "paypal"): 1.0})
    d = evaluate(vals, _captures_ok(), checks, trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.MISSING_VISION_CONFIDENCE.value


def test_model_call_missing_and_capture_missing():
    """Cache miss = model/API failure; missing capture likewise permanent."""
    vals = _valid_panel()
    vals[1] = AgentValidity("agent_b_text", "qwen-2.5-72b", False, raw_present=False)
    outs = [_out("agent_a_text", [_brand()]), _out("agent_b_text", [_brand()]),
            _out(VISION, [_brand()])]
    checks = _ground(build_cue_checks(outs, T_LOGO), {("brand_claim", "paypal"): 1.0})
    d = evaluate(vals, {"screenshot": False, "html": True}, checks,
                 trust_calibrated=1.0, tau=TAU)
    assert d.action == "abstain"
    assert d.primary_reason == Reason.MODEL_CALL_MISSING.value
    assert f"{Reason.MISSING_CAPTURE.value}:screenshot" in d.invalid_reasons


def test_clean_all_pass_acts_only_if_confidence_passes():
    """The all-pass case: acts at trust >= tau, abstains (score_below_tau) below it."""
    outs = [_out("agent_a_text", [_brand(), _logo()]),
            _out("agent_b_text", [_brand(), _logo()]),
            _out(VISION, [_brand(), _logo()])]
    scores = {("brand_claim", "paypal"): 1.0, ("logo_brand", "paypal"): T_LOGO + 0.01}

    d_hi = evaluate(_valid_panel(), _captures_ok(),
                    _ground(build_cue_checks(outs, T_LOGO), scores),
                    trust_calibrated=TAU, tau=TAU)
    assert d_hi.action == "act"
    assert d_hi.primary_reason == Reason.ACT.value

    d_lo = evaluate(_valid_panel(), _captures_ok(),
                    _ground(build_cue_checks(outs, T_LOGO), scores),
                    trust_calibrated=TAU - 1e-6, tau=TAU)
    assert d_lo.action == "abstain"
    assert d_lo.primary_reason == Reason.SCORE_BELOW_TAU.value


def test_logo_threshold_is_the_frozen_one():
    """A majority logo cue passes exactly at the frozen perceptual threshold."""
    outs = [_out("agent_a_text", [_logo()]), _out("agent_b_text", [_logo()]),
            _out(VISION, [_logo(), _brand()])]
    below = _ground(build_cue_checks(outs, T_LOGO), {("logo_brand", "paypal"): T_LOGO - 0.01,
                                                     ("brand_claim", "paypal"): 1.0})
    at = _ground(build_cue_checks(outs, T_LOGO), {("logo_brand", "paypal"): T_LOGO,
                                                  ("brand_claim", "paypal"): 1.0})
    d_below = evaluate(_valid_panel(), _captures_ok(), below, trust_calibrated=1.0, tau=TAU)
    d_at = evaluate(_valid_panel(), _captures_ok(), at, trust_calibrated=1.0, tau=TAU)
    assert d_below.action == "abstain" and d_below.primary_reason == Reason.CHECK_FAILED.value
    assert d_at.action == "act"
