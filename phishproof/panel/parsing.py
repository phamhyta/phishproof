"""Turn a model's raw text into a ``MemberOutput``.

Parsing works at the verdict level. A response that fails validation still votes if its verdict
can be read, but it carries no cues and no confidence. A response with no readable verdict is a
failure and never votes.
"""

from __future__ import annotations

import re

from pydantic import ValidationError

from ..schema import Cue, Label, MemberOutput
from .prompts import RawAgentResponse

_VERDICT = re.compile(r'"verdict"\s*:\s*"(phish|benign)"', re.IGNORECASE)


def _json_body(raw: str) -> str:
    text = raw.strip()
    if text.startswith("{"):
        return text
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else "{}"


def _usable_confidence(value: float | None) -> float | None:
    return value if value is not None and 0.0 <= value <= 1.0 else None


def parse_response(member_id: str, raw: str) -> MemberOutput:
    try:
        parsed = RawAgentResponse.model_validate_json(_json_body(raw))
    except ValidationError:
        match = _VERDICT.search(raw)
        if match is None:
            return MemberOutput(member_id=member_id, raw_response=raw, failure="no_verdict")
        return MemberOutput(
            member_id=member_id,
            verdict=Label(match.group(1).lower()),
            raw_response=raw,
            failure="invalid_schema",
        )
    cues = [
        Cue(type=cue.type, value=cue.value, raw_value=cue.value, asserted_by=member_id)
        for cue in parsed.cues
    ]
    return MemberOutput(
        member_id=member_id,
        verdict=parsed.verdict,
        cues=cues,
        confidence=_usable_confidence(parsed.confidence),
        raw_response=raw,
    )
