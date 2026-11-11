"""B5, a separate model call that reads every verdict and cited cue and returns a final verdict."""

from __future__ import annotations

import json
from collections import Counter

from ..config import MemberConfig
from ..schema import Label, MemberOutput
from .scores import majority_confidence

CONSOLIDATOR_SYSTEM = """You are the CONSOLIDATOR of a multi-agent phishing detector. \
Several analyst agents have each independently judged whether a web page is phishing and \
listed the evidence they used. Your job is to deliberate over their combined reasoning — \
how much they agree, how specific and strong each agent's evidence is, and whether the \
evidence actually supports the verdict — and then output a final verdict plus a CALIBRATED \
confidence in [0,1] that the verdict is correct (0.5 = a coin toss, 1.0 = certain). \
Reserve high confidence for cases where the agents agree on strong, specific evidence. \
Output STRICT JSON only: {"verdict": "phish"|"benign", "confidence": <0..1>}."""


def _render_agents(outputs: list[MemberOutput]) -> str:
    lines = []
    for o in outputs:
        cues = "; ".join(f"{c.type.value}={c.value}" for c in o.cues) or "(no cues)"
        conf = f"{o.confidence:.2f}" if o.confidence is not None else "n/a"
        lines.append(f"- {o.member_id}: verdict={o.verdict.value if o.verdict else 'none'} (self-conf {conf}); evidence: {cues}")
    return "\n".join(lines)


class Consolidator:
    def __init__(self, client, config: MemberConfig) -> None:
        self.client = client
        self.config = config

    def score(self, outputs: list[MemberOutput]) -> tuple[Label, float]:
        votes = Counter(o.verdict for o in outputs if o.verdict is not None)
        if not votes:
            return Label.BENIGN, 0.0
        user = (
            "Consolidate these analysts into a final verdict + confidence.\n\n"
            + _render_agents(outputs)
            + '\n\nReturn JSON: {"verdict": "phish"|"benign", "confidence": 0..1}'
        )
        raw = self.client.complete_json(self.config, CONSOLIDATOR_SYSTEM, user)
        try:
            parsed = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            return Label(parsed["verdict"]), max(0.0, min(1.0, float(parsed["confidence"])))
        except (KeyError, TypeError, ValueError):
            return votes.most_common(1)[0][0], majority_confidence(outputs)
