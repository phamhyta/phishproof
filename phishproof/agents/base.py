"""Agent interface: a tool-using LLM that inspects one page."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .._release import pending
from ..schema import AgentOutput
from ..types import PageRecord

if TYPE_CHECKING:
    from .page_context import PageContext


class Agent(ABC):
    """Base class for a panel agent.

    An agent plans what to examine, calls verification tools, and records typed
    cues in a shared evidence set before returning a structured verdict.
    """

    agent_id: str
    modality: str

    @abstractmethod
    def inspect(self, page: PageRecord, context: "PageContext") -> AgentOutput:
        """Inspect a page and return a structured verdict with cited cues."""


class ToolUsingAgent(Agent):
    """An agent that reasons in a plan/act loop over the tool registry."""

    def __init__(self, agent_id: str, modality: str) -> None:
        self.agent_id = agent_id
        self.modality = modality

    def inspect(self, page: PageRecord, context: "PageContext") -> AgentOutput:
        pending("agents.base.ToolUsingAgent.inspect")
