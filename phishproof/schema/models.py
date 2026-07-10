"""Evidence and scoring data models.

These frozen records are the internal data contracts that flow through the
pipeline. They are fully defined here because they carry only structure, not
method logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Label
from .cue_types import CueType


@dataclass(frozen=True)
class Cue:
    """A single typed, checkable evidence cue cited by one agent.

    ``value`` is the normalized claim (for example, a form-action domain or a
    brand name) that a verification tool can re-derive from the page.
    """

    type: CueType
    value: str
    rationale: str = ""


@dataclass(frozen=True)
class AgentOutput:
    """One agent's structured verdict for a page."""

    agent_id: str
    label: Label
    confidence: float | None
    cues: tuple[Cue, ...] = ()


@dataclass(frozen=True)
class GroundingResult:
    """Result of re-deriving one agreed cue from the page with a tool."""

    cue: Cue
    grounded: bool
    tool: str
    detail: str = ""


@dataclass(frozen=True)
class TrustScore:
    """The trust signal for one page and its decomposition."""

    raw: float
    calibrated: float
    per_type_agreement: dict[str, float] = field(default_factory=dict)
    grounded: bool = True


@dataclass
class EvidenceBundle:
    """Everything gathered for one page: panel outputs, grounding, and trust."""

    page_id: str
    outputs: tuple[AgentOutput, ...] = ()
    grounding: tuple[GroundingResult, ...] = ()
    trust: TrustScore | None = None
