"""B6 — MultiPhishGuard (multiphishguard2025): a multi-agent detector whose consolidated-
rationale confidence is the trust score.

Two implementations:
  - b6_multiphishguard_proxy: cheap stand-in = mean confidence of the majority-agreeing
    agents. Flagged as a proxy (it leans on the agents' self-reported confidence).
  - MultiPhishGuardConsolidator: faithful version — a separate consolidator LLM reads every
    agent's verdict + cited evidence and deliberates a final verdict and a calibrated
    confidence, the way MultiPhishGuard consolidates its agents' rationales.
"""

from __future__ import annotations

from collections import Counter

from ..config import AgentConfig
from ..schema import AgentOutput, Label

# ---- cheap proxy (kept for ablation / fallback) ----


def b6_multiphishguard_proxy(outputs: list[AgentOutput]) -> float:
    if not outputs:
        return 0.0
    majority = Counter(o.verdict for o in outputs).most_common(1)[0][0]
    agree = [o for o in outputs if o.verdict == majority]
    confs = [o.confidence for o in agree if o.confidence is not None]
    if not confs:
        return len(agree) / len(outputs)
    return float(sum(confs) / len(confs))


# ---- faithful consolidator ----

CONSOLIDATOR_SYSTEM = """You are the CONSOLIDATOR of a multi-agent phishing detector. \
Several analyst agents have each independently judged whether a web page is phishing and \
listed the evidence they used. Your job is to deliberate over their combined reasoning — \
how much they agree, how specific and strong each agent's evidence is, and whether the \
evidence actually supports the verdict — and then output a final verdict plus a CALIBRATED \
confidence in [0,1] that the verdict is correct (0.5 = a coin toss, 1.0 = certain). \
Reserve high confidence for cases where the agents agree on strong, specific evidence. \
Output STRICT JSON only: {"verdict": "phish"|"benign", "confidence": <0..1>}."""


def _render_agents(outputs: list[AgentOutput]) -> str:
    lines = []
    for o in outputs:
        cues = "; ".join(f"{c.type.value}={c.value}" for c in o.cues) or "(no cues)"
        conf = f"{o.confidence:.2f}" if o.confidence is not None else "n/a"
        lines.append(f"- {o.agent_id}: verdict={o.verdict.value} (self-conf {conf}); evidence: {cues}")
    return "\n".join(lines)


class MultiPhishGuardConsolidator:
    """Faithful B6: consolidate agents' rationales via a separate LLM judge."""

    def __init__(self, client, cfg: AgentConfig) -> None:
        self.client = client
        self.cfg = cfg

    def score(self, outputs: list[AgentOutput]) -> tuple[Label, float]:
        if not outputs:
            return Label.BENIGN, 0.0
        user = ("Consolidate these analysts into a final verdict + confidence.\n\n"
                + _render_agents(outputs)
                + '\n\nReturn JSON: {"verdict": "phish"|"benign", "confidence": 0..1}')
        raw = self.client.complete_json(self.cfg, CONSOLIDATOR_SYSTEM, user)
        try:
            import json
            txt = raw[raw.find("{"): raw.rfind("}") + 1]
            d = json.loads(txt)
            verdict = Label(d.get("verdict", "benign"))
            conf = float(d.get("confidence", 0.5))
            return verdict, max(0.0, min(1.0, conf))
        except Exception:  # noqa: BLE001
            return b6_majority_label(outputs), b6_multiphishguard_proxy(outputs)


def b6_majority_label(outputs: list[AgentOutput]) -> Label:
    return Counter(o.verdict for o in outputs).most_common(1)[0][0]
