"""Deterministic mock agent — exercises the panel + aggregation without any model.

Builds plausible cues from the real page (via the grounding tools), then applies
per-agent drop/override so different mock agents produce overlapping-but-not-identical
cue sets — enough to test consensus, agreement, and grounding end to end.
"""

from __future__ import annotations

from ..schema import AgentOutput, Cue, CueType, Label, PageRecord
from ..tools.dom import form_action_domains, has_credential_intent


class MockAgent:
    def __init__(
        self,
        agent_id: str,
        drop: set[CueType] | None = None,
        override: dict[CueType, str] | None = None,
        verdict: Label = Label.PHISH,
    ) -> None:
        self.agent_id = agent_id
        self.drop = drop or set()
        self.override = override or {}
        self.verdict = verdict

    def _cue(self, ctype: CueType, value: str | None) -> Cue | None:
        if ctype in self.drop or not value:
            return None
        value = self.override.get(ctype, value)
        return Cue(type=ctype, value=value, raw_value=value, asserted_by=self.agent_id)

    def analyze(self, page: PageRecord) -> AgentOutput:
        fads = sorted(form_action_domains(page))
        cred = "yes" if has_credential_intent(page) else "no"
        candidates = [
            self._cue(CueType.BRAND_CLAIM, page.brand),
            self._cue(CueType.FORM_ACTION_DOMAIN, fads[0] if fads else None),
            self._cue(CueType.CREDENTIAL_INTENT, cred),
            self._cue(CueType.LOGO_BRAND, page.brand),
        ]
        cues = [c for c in candidates if c is not None]
        return AgentOutput(agent_id=self.agent_id, verdict=self.verdict, cues=cues,
                           confidence=0.9)
