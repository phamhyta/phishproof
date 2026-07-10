"""Public input/output data types for PhishProof.

These are the stable data contracts a caller sees at the edges of the pipeline:
a :class:`PageRecord` goes in, a :class:`Prediction` comes out. They are fully
implemented here because they carry no method logic -- only the shape of the
data. The reasoning that turns one into the other is withheld until release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["phishing", "benign", "unknown"]
Decision = Literal["act", "abstain"]


@dataclass(frozen=True)
class PageRecord:
    """A single page to be judged.

    Attributes
    ----------
    id:
        Stable page identifier.
    url:
        Page URL (used by the URL and redirect verifiers).
    title:
        Optional page title.
    html_path:
        Optional path to saved rendered HTML (used by the DOM/brand verifiers).
    screenshot_path:
        Optional path to a screenshot (used by the vision agent and logo tool).
    metadata:
        Optional free-form caller notes carried through to the output.
    """

    id: str
    url: str
    title: str = ""
    html_path: str | None = None
    screenshot_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "PageRecord":
        page_id = str(row.get("id") or row.get("page_id") or "").strip()
        url = str(row.get("url") or "").strip()
        if not page_id:
            raise ValueError("page record is missing 'id'")
        if not url:
            raise ValueError(f"page record {page_id!r} is missing 'url'")
        return cls(
            id=page_id,
            url=url,
            title=str(row.get("title") or ""),
            html_path=_optional_str(row.get("html_path")),
            screenshot_path=_optional_str(row.get("screenshot_path")),
            metadata=dict(row.get("metadata") or {}),
        )


@dataclass(frozen=True)
class Prediction:
    """The selective verdict returned for one page.

    Attributes
    ----------
    id:
        Page identifier echoed from the input.
    label:
        The panel verdict (``phishing`` / ``benign`` / ``unknown``).
    confidence:
        Calibrated probability that ``label`` is correct (the trust score).
    decision:
        ``act`` if the calibrated trust clears the operating threshold, else
        ``abstain`` (escalate to review).
    reasons:
        The agreed, page-grounded evidence cues that justify the verdict -- the
        checkable explanation returned when the detector acts.
    """

    id: str
    label: Label
    confidence: float
    decision: Decision = "abstain"
    reasons: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
