"""Prompt construction for the panel agents.

The prompt templates are part of the reference implementation and are released
with the paper; only their signatures are public here.
"""

from __future__ import annotations

from .._release import pending
from ..types import PageRecord


def vision_agent_prompt(page: PageRecord) -> str:
    """Build the instruction shown to the screenshot-reading agent."""
    pending("agents.prompts.vision_agent_prompt")


def text_agent_prompt(page: PageRecord, dom_text: str) -> str:
    """Build the instruction shown to a structure-reading agent."""
    pending("agents.prompts.text_agent_prompt")


def cue_schema_instructions() -> str:
    """Return the typed-cue output-format instructions shared by all agents."""
    pending("agents.prompts.cue_schema_instructions")
