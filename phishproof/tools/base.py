"""Verifier interface and registry.

A verifier is a tool that re-derives one cue type from the page so an agreed cue
can be grounded. The registry plumbing is public; each verifier body is released
with the paper.
"""

from __future__ import annotations

from typing import Protocol

from ..schema import Cue, GroundingResult
from ..types import PageRecord


class Verifier(Protocol):
    """A grounding tool: given a cited cue and the page, re-derive it."""

    name: str

    def verify(self, cue: Cue, page: PageRecord) -> GroundingResult: ...


class ToolRegistry:
    """Maps verifier names to the tool that grounds the corresponding cue."""

    def __init__(self) -> None:
        self._by_name: dict[str, Verifier] = {}

    def register(self, tool: Verifier) -> None:
        self._by_name[tool.name] = tool

    def get(self, name: str) -> Verifier:
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name)
