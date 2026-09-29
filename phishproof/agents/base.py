"""Agent interface + the LLM-backed evidence agent (C1).

An evidence agent reads the page, decides phish/benign, and returns typed cues from
the fixed schema. Real agents call a model via ChatClient; MockAgent (mock.py) returns
deterministic output so the panel/aggregation can be tested without any model.
"""

from __future__ import annotations

import json
from typing import Protocol

from ..config import AgentConfig
from ..schema import AgentOutput, Cue, PageRecord
from .client import ChatClient
from .page_context import render_context
from .prompts import SYSTEM_PROMPT, RawAgentResponse, build_user_prompt


class Agent(Protocol):
    agent_id: str

    def analyze(self, page: PageRecord) -> AgentOutput: ...


def _parse(agent_id: str, raw: str) -> AgentOutput:
    """Validate the model's JSON into an AgentOutput; tolerate junk around the JSON."""
    text = raw.strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        text = text[i : j + 1] if i != -1 and j != -1 else "{}"
    try:
        parsed = RawAgentResponse.model_validate_json(text)
    except Exception:  # noqa: BLE001 - malformed model output -> empty, low-trust cue set
        try:
            data = json.loads(text)
            parsed = RawAgentResponse.model_validate(data)
        except Exception:  # noqa: BLE001
            from ..schema import Label
            return AgentOutput(agent_id=agent_id, verdict=Label.BENIGN, cues=[],
                               raw_response=raw)
    cues = [Cue(type=c.type, value=c.value, raw_value=c.value, asserted_by=agent_id)
            for c in parsed.cues]
    return AgentOutput(agent_id=agent_id, verdict=parsed.verdict, cues=cues,
                       confidence=parsed.confidence, raw_response=raw)


class EvidenceAgent:
    """LLM-backed agent. Works for local (Ollama) and API models via AgentConfig."""

    def __init__(self, cfg: AgentConfig, client: ChatClient | None = None) -> None:
        self.cfg = cfg
        self.agent_id = cfg.id
        self.client = client or ChatClient()

    def analyze(self, page: PageRecord) -> AgentOutput:
        context = render_context(page)
        user = build_user_prompt(context)
        image = page.screenshot_path if self.cfg.modality == "vision" else None
        raw = self.client.complete_json(self.cfg, SYSTEM_PROMPT, user, image)
        return _parse(self.agent_id, raw)
