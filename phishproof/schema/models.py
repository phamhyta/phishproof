"""Records passed between the layers of the pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .cue_types import GATING_CUE_TYPES, CueType


class Label(str, Enum):
    PHISH = "phish"
    BENIGN = "benign"


class Cue(BaseModel):
    """One typed claim about a page."""

    type: CueType
    value: str
    raw_value: str | None = None
    asserted_by: str | None = None

    @field_validator("value")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def key(self) -> tuple[str, str]:
        return (self.type.value, self.value.lower())


class MemberOutput(BaseModel):
    """What one panel member returned for a page.

    ``verdict`` is None when the response has no recoverable verdict. Such a member never
    votes. ``confidence`` is None unless the JSON field parsed.
    """

    member_id: str
    verdict: Label | None = None
    cues: list[Cue] = Field(default_factory=list)
    confidence: float | None = None
    raw_response: str | None = None
    failure: str | None = None
    served_model: str | None = None
    provider: str | None = None


class PageRecord(BaseModel):
    """A captured page. ``label`` is unknown outside labelled corpora."""

    page_id: str
    url: str
    label: Label | None = None
    dom_html_path: str | None = None
    screenshot_path: str | None = None
    raw_dir: str | None = None
    brand: str | None = None
    final_url: str | None = None
    certificate_org: str | None = None
    redirect_chain: list[str] | None = None
    source: str | None = None
    split: str | None = None

    @property
    def host_url(self) -> str:
        return self.final_url or self.url


class GroundingResult(BaseModel):
    """A verifier's score for one cue."""

    cue: Cue
    score: float
    tool: str | None = None


class CueCheck(BaseModel):
    """Verification record of one consensus cue. ``passed`` is None when unavailable."""

    type: CueType
    value: str
    supporters: list[str] = Field(default_factory=list)
    tool: str | None = None
    score: float | None = None
    threshold: float = 1.0
    passed: bool | None = None

    @property
    def available(self) -> bool:
        return self.score is not None

    @property
    def gating(self) -> bool:
        return self.type in GATING_CUE_TYPES


class PageEvidence(BaseModel):
    """Everything the decision policy reads for a page, in a form that can be stored and replayed."""

    page_id: str
    url: str
    label: Label | None = None
    outputs: list[MemberOutput] = Field(default_factory=list)
    verdict: Label | None = None
    rank: float | None = None
    complete: bool = False
    gea: float = 0.0
    groundedness: float = 0.0
    checks: list[CueCheck] = Field(default_factory=list)
    named_brands: list[str] = Field(default_factory=list)
    bound: bool = False

    @property
    def gating_checks(self) -> list[CueCheck]:
        return [check for check in self.checks if check.gating]

    @property
    def gate_passed(self) -> bool:
        gating = self.gating_checks
        return bool(gating) and all(check.passed is True for check in gating)
