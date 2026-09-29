"""The evidence-agent panel (C1): run M agents, collect the shared cue set + majority label."""

from __future__ import annotations

from collections import Counter

from ..config import AgentConfig, PanelConfig, load_panel
from ..schema import AgentOutput, Label, PageRecord
from .base import Agent, EvidenceAgent
from .client import ChatClient


class Panel:
    def __init__(self, agents: list[Agent]) -> None:
        if not agents:
            raise ValueError("panel needs at least one agent")
        self.agents = agents

    @classmethod
    def from_config(cls, cfg: PanelConfig | None = None, client: ChatClient | None = None) -> "Panel":
        cfg = cfg or load_panel()
        client = client or ChatClient()
        return cls([EvidenceAgent(a, client) for a in cfg.panel])

    @classmethod
    def from_agent_configs(cls, configs: list[AgentConfig], client: ChatClient | None = None) -> "Panel":
        client = client or ChatClient()
        return cls([EvidenceAgent(a, client) for a in configs])

    def run(self, page: PageRecord) -> list[AgentOutput]:
        return [a.analyze(page) for a in self.agents]

    def run_batched(self, pages, progress=None) -> dict[str, list[AgentOutput]]:
        """Run agent-by-agent over all pages so each model is loaded once (16 GB-friendly).

        For each agent we sweep every page (model stays resident), then move to the next
        agent. Returns {page_id: [out_agent0, out_agent1, ...]} in panel order. A failed
        call becomes an empty-cue BENIGN output so the page keeps a fixed panel size M.
        """
        outputs_by_page: dict[str, list[AgentOutput]] = {p.page_id: [] for p in pages}
        for agent in self.agents:
            for i, p in enumerate(pages, 1):
                try:
                    out = agent.analyze(p)
                except Exception:  # noqa: BLE001 - one bad call must not abort the sweep
                    out = AgentOutput(agent_id=agent.agent_id, verdict=Label.BENIGN, cues=[])
                outputs_by_page[p.page_id].append(out)
                if progress is not None:
                    progress(agent.agent_id, i, len(pages))
        return outputs_by_page

    @staticmethod
    def majority_label(outputs: list[AgentOutput]) -> Label:
        counts = Counter(o.verdict for o in outputs)
        # Tie -> phish (fail safe toward flagging); deterministic on ties.
        phish = counts.get(Label.PHISH, 0)
        benign = counts.get(Label.BENIGN, 0)
        return Label.PHISH if phish >= benign else Label.BENIGN
