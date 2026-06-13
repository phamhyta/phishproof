from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Label = Literal["phishing", "benign", "unknown"]


@dataclass(frozen=True)
class PageRecord:
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
    id: str
    label: Label
    confidence: float
    reasons: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "reasons": list(self.reasons),
        }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
