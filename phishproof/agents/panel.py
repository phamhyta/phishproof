"""The cross-modal agent panel."""

from __future__ import annotations

from collections.abc import Sequence

from .._release import pending
from ..config import PanelConfig
from ..schema import AgentOutput
from ..types import PageRecord
from .base import Agent


class AgentPanel:
    """A small, cross-modal panel: one vision agent plus capable text agents.

    Diversity across modality -- not the number of agents -- supplies the
    agreement signal, so the panel is kept deliberately small: every added agent
    costs another round of model and tool calls.
    """

    def __init__(self, agents: Sequence[Agent], config: PanelConfig | None = None) -> None:
        self.agents = tuple(agents)
        self.config = config or PanelConfig()

    @classmethod
    def from_config(cls, config: PanelConfig) -> "AgentPanel":
        """Instantiate the configured vision and text agents."""
        pending("agents.panel.AgentPanel.from_config")

    def inspect(self, page: PageRecord) -> list[AgentOutput]:
        """Run every agent over the page and collect their structured verdicts."""
        pending("agents.panel.AgentPanel.inspect")
