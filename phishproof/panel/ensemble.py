"""Query every panel member on a page and take the majority verdict."""

from __future__ import annotations

import logging
from collections import Counter

from ..config import PanelConfig, load_panel
from ..schema import Label, MemberOutput, PageRecord
from .client import ChatClient
from .member import LLMMember, PanelMember

log = logging.getLogger(__name__)


def majority_verdict(outputs: list[MemberOutput]) -> Label | None:
    """Majority of the members that voted. A tied vote resolves to phishing."""
    votes = Counter(output.verdict for output in outputs if output.verdict is not None)
    if not votes:
        return None
    return Label.PHISH if votes[Label.PHISH] >= votes[Label.BENIGN] else Label.BENIGN


class Panel:
    def __init__(self, members: list[PanelMember]) -> None:
        if not members:
            raise ValueError("a panel needs at least one member")
        self.members = members

    @classmethod
    def from_config(cls, config: PanelConfig | None = None, client: ChatClient | None = None):
        config = config or load_panel()
        client = client or ChatClient()
        return cls([LLMMember(member, client) for member in config.panel])

    def run(self, page: PageRecord) -> list[MemberOutput]:
        return [self._query(member, page) for member in self.members]

    @staticmethod
    def _query(member: PanelMember, page: PageRecord) -> MemberOutput:
        try:
            return member.analyze(page)
        except Exception as error:  # a failed call must not stop the other members
            log.warning("%s failed on %s: %s", member.member_id, page.page_id, error)
            name = type(error).__name__
            failure = "timeout" if "timeout" in name.lower() else name
            return MemberOutput(member_id=member.member_id, failure=failure)
