"""B7 -- PhishDebate (phishdebate2025): a role-based multi-agent debate whose moderator
resolves the deliberated rationale into a final verdict + confidence; that confidence is the
selective trust score.

Difference from B6 (MultiPhishGuard): B6 consolidates the panel's round-0 positions in a
single pass. PhishDebate runs R rounds of cross-examination first -- each agent sees its
peers' arguments and may revise its verdict and reasoning -- and only then does a moderator
resolve the (possibly shifted) debate into a confident answer. The same M=3 panel and the
same grounded page context are reused, so the comparison isolates *deliberation*
(debate-then-resolve) from B6's single-pass consolidation and from PhishProof's
no-deliberation per-type agreement.

The debate rounds are TEXT calls: each agent's round-0 visual/textual evidence is already
captured as its cited cues, so the debaters argue over that grounded evidence (and the
compact page context) rather than re-reading the screenshot every round.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..agents.client import ChatClient
from ..agents.page_context import render_context
from ..config import AgentConfig
from ..schema import AgentOutput, Label, PageRecord

DEBATE_SYSTEM = """You are one of several security analysts debating whether a web page is \
PHISHING or LEGITIMATE (benign). Decide by whether the brand the page presents is consistent \
with the domain it runs on: a recognizable brand on a NON-official / mismatched domain \
(especially with a credential form) is phishing; a brand on its own official domain, or a page \
with no impersonated brand at all, is benign. You can see the page context and your fellow \
analysts' current positions. Weigh their arguments honestly -- change your verdict if their \
evidence is stronger, hold it if it is not -- and give a one-sentence argument. \
Output STRICT JSON only: \
{"verdict":"phish"|"benign","argument":"<one sentence>","confidence":<0..1>}."""

MODERATOR_SYSTEM = """You are the MODERATOR of a multi-analyst phishing debate. The analysts \
have argued, seen each other's reasoning, and revised their positions. Read the page context \
and their final positions, judge how strongly they converge on specific, page-grounded \
evidence (brand-vs-domain mismatch, form-action domain, credential intent, logo), and output a \
final verdict plus a CALIBRATED confidence in [0,1] that the verdict is correct (0.5 = a coin \
toss, 1.0 = certain). Reserve high confidence for genuine convergence on strong evidence; lower \
it when the debate stays split or rests on weak cues. \
Output STRICT JSON only: {"verdict":"phish"|"benign","confidence":<0..1>}."""


@dataclass
class Position:
    agent_id: str
    verdict: Label
    argument: str
    confidence: float | None = None


def _cues_to_argument(out: AgentOutput) -> str:
    if not out.cues:
        return "no specific evidence cited"
    return "; ".join(f"{c.type.value}={c.value}" for c in out.cues)


def _render_positions(positions: list[Position], exclude: str | None = None) -> str:
    lines = []
    for p in positions:
        if p.agent_id == exclude:
            continue
        conf = f"{p.confidence:.2f}" if p.confidence is not None else "n/a"
        lines.append(f"- {p.agent_id}: {p.verdict.value} (conf {conf}) -- {p.argument}")
    return "\n".join(lines) or "(no other analysts)"


def _parse(raw: str) -> tuple[Label, str, float | None]:
    """Lenient parse of a debate/moderator JSON reply -> (verdict, argument, confidence)."""
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        verdict = Label(str(d.get("verdict", "benign")).lower())
        arg = str(d.get("argument", ""))[:300]
        c = d.get("confidence")
        conf = max(0.0, min(1.0, float(c))) if c is not None else None
        return verdict, arg, conf
    except Exception:  # noqa: BLE001 - malformed output -> abstain-ish default
        return Label.BENIGN, "", None


class PhishDebate:
    """B7: R rounds of simultaneous cross-examination over the panel's positions, then a
    moderator resolves the debate into a final verdict + confidence (the trust score)."""

    def __init__(self, client: ChatClient, debater_cfgs: list[AgentConfig],
                 moderator_cfg: AgentConfig, rounds: int = 2) -> None:
        self.client = client
        self.debaters = {c.id: c for c in debater_cfgs}
        self.moderator_cfg = moderator_cfg
        self.rounds = rounds

    def _revise(self, cfg: AgentConfig, context: str, positions: list[Position],
                agent_id: str) -> tuple[Label, str, float | None]:
        peers = _render_positions(positions, exclude=agent_id)
        user = (f"{context}\n\nOther analysts' current positions:\n{peers}\n\n"
                f"You are {agent_id}. Reconsider in light of their arguments and return your "
                f"revised verdict, a one-sentence argument, and confidence. JSON only: "
                '{"verdict":"phish"|"benign","argument":"<one sentence>","confidence":<0..1>}')
        return _parse(self.client.complete_json(cfg, DEBATE_SYSTEM, user, image_path=None))

    def score(self, outputs: list[AgentOutput], page: PageRecord) -> tuple[Label, float]:
        """Run the debate and return (moderator verdict, confidence in [0,1])."""
        if not outputs:
            return Label.BENIGN, 0.0
        context = render_context(page)
        positions = [Position(o.agent_id, o.verdict, _cues_to_argument(o), o.confidence)
                     for o in outputs]

        for _ in range(self.rounds):
            snapshot = positions  # simultaneous: every agent sees the prior round's snapshot
            updated: list[Position] = []
            for p in snapshot:
                cfg = self.debaters.get(p.agent_id)
                if cfg is None:               # agent not in debater set -> keep its position
                    updated.append(p)
                    continue
                v, arg, conf = self._revise(cfg, context, snapshot, p.agent_id)
                updated.append(Position(p.agent_id, v, arg or p.argument,
                                        conf if conf is not None else p.confidence))
            positions = updated

        transcript = _render_positions(positions)
        user = ("Resolve this phishing debate into a final verdict + confidence.\n\n"
                f"{context}\n\nFinal analyst positions:\n{transcript}\n\n"
                'Return JSON: {"verdict":"phish"|"benign","confidence":0..1}')
        verdict, _, conf = _parse(self.client.complete_json(
            self.moderator_cfg, MODERATOR_SYSTEM, user, image_path=None))
        return verdict, conf if conf is not None else 0.5
