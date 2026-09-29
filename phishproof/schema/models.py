"""Core data models — the contract every component reads/writes.

Flow: PageRecord -> (agents) AgentOutput -> (aggregate) GEAResult.
Cues are constrained to the schema in cue_types.py.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .cue_types import CueType


class Label(str, Enum):
    PHISH = "phish"
    BENIGN = "benign"


class Cue(BaseModel):
    """A single typed, tool-checkable evidence claim (t, v)."""

    type: CueType
    value: str                       # normalized value (eTLD+1, canonical brand, "yes"/"no")
    raw_value: str | None = None     # original value before normalization
    asserted_by: str | None = None   # agent id that raised this cue

    @field_validator("value")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def key(self) -> tuple[str, str]:
        """Normalized identity used for consensus matching across agents."""
        return (self.type.value, self.value.lower())


class AgentOutput(BaseModel):
    """One evidence agent's verdict + grounded cues (eq:output)."""

    agent_id: str
    verdict: Label
    cues: list[Cue] = Field(default_factory=list)
    confidence: float | None = None   # optional self-reported confidence (for baselines)
    raw_response: str | None = None    # raw model text, logged for audit


class PageRecord(BaseModel):
    """A captured page. cert/redirect may be None on the Phishpedia dataset."""

    page_id: str
    url: str
    label: Label
    dom_html_path: str | None = None
    screenshot_path: str | None = None
    raw_dir: str | None = None            # source folder (aux files: logo bbox, etc.)
    brand: str | None = None              # dataset brand annotation (gold), if present
    final_url: str | None = None
    certificate_org: str | None = None    # None on Phishpedia (decision 0b)
    redirect_chain: list[str] | None = None
    source: str | None = None             # e.g. "phishpedia", "phishtank"
    split: str | None = None              # "calibration" | "test"


class GroundingResult(BaseModel):
    """Tool output for one consensus cue (eq:ground)."""

    cue: Cue
    score: float                          # in [0,1]; structural tools return 0/1
    tool: str | None = None               # which tool produced the score


class GEAResult(BaseModel):
    """End-to-end result for one page (Algorithm 1)."""

    page_id: str
    verdict: Label                        # majority label over the panel
    agreement: float                      # A = |consensus| / |all cues|
    groundedness: float                   # G = mean tool score over consensus
    gea: float                            # A * G
    calibrated_trust: float | None = None  # kappa(gea)
    consensus_cues: list[Cue] = Field(default_factory=list)
    acted: bool | None = None             # True = act, False = abstain
