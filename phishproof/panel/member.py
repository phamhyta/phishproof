"""A panel member: one model queried once per page with the fixed prompt."""

from __future__ import annotations

from typing import Protocol

from ..config import MemberConfig
from ..schema import MemberOutput, PageRecord
from .client import ChatClient, ReplyDetails
from .context import render_context
from .parsing import parse_response
from .prompts import SYSTEM_PROMPT, build_user_prompt


class PanelMember(Protocol):
    member_id: str

    def analyze(self, page: PageRecord) -> MemberOutput: ...


def _failure_cause(output: MemberOutput, details: ReplyDetails) -> str | None:
    if details.refusal:
        return "refusal"
    if details.finish_reason == "length" and output.confidence is None:
        return "truncated"
    return output.failure


class LLMMember:
    def __init__(self, config: MemberConfig, client: ChatClient | None = None) -> None:
        self.config = config
        self.member_id = config.id
        self.client = client or ChatClient()

    def analyze(self, page: PageRecord) -> MemberOutput:
        user = build_user_prompt(render_context(page))
        image = page.screenshot_path if self.config.modality == "vision" else None
        reply = self.client.complete(self.config, SYSTEM_PROMPT, user, image)
        output = parse_response(self.member_id, reply.text)
        return output.model_copy(
            update={
                "failure": _failure_cause(output, reply.details),
                "served_model": reply.details.served_model,
                "provider": reply.details.provider,
            }
        )
