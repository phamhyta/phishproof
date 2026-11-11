"""A model-free panel member, useful for trying the pipeline without API keys."""

from __future__ import annotations

from ..schema import Cue, CueType, Label, MemberOutput, PageRecord
from ..verification.dom import form_action_domains, has_credential_intent


class MockMember:
    def __init__(
        self,
        member_id: str,
        verdict: Label = Label.PHISH,
        confidence: float | None = 0.9,
        drop: set[CueType] | None = None,
        override: dict[CueType, str] | None = None,
    ) -> None:
        self.member_id = member_id
        self.verdict = verdict
        self.confidence = confidence
        self.drop = drop or set()
        self.override = override or {}

    def _cue(self, cue_type: CueType, value: str | None) -> Cue | None:
        value = self.override.get(cue_type, value)
        if cue_type in self.drop or not value:
            return None
        return Cue(type=cue_type, value=value, raw_value=value, asserted_by=self.member_id)

    def analyze(self, page: PageRecord) -> MemberOutput:
        forms = sorted(form_action_domains(page))
        candidates = [
            self._cue(CueType.BRAND_CLAIM, page.brand),
            self._cue(CueType.FORM_ACTION_DOMAIN, forms[0] if forms else None),
            self._cue(CueType.CREDENTIAL_INTENT, "yes" if has_credential_intent(page) else "no"),
            self._cue(CueType.LOGO_BRAND, page.brand),
        ]
        return MemberOutput(
            member_id=self.member_id,
            verdict=self.verdict,
            cues=[cue for cue in candidates if cue is not None],
            confidence=self.confidence,
        )
