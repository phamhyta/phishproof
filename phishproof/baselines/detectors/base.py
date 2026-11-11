"""Types shared by the dedicated detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ...schema import Label, PageRecord

PHISH = "phish"
BENIGN = "benign"


@dataclass(frozen=True)
class Page:
    page_id: str
    url: str
    html_path: Path | None
    screenshot_path: Path | None
    label: int = -1
    split: str = ""
    group_id: str = ""

    @classmethod
    def from_record(cls, record: PageRecord, group_id: str = "") -> Page:
        label = {Label.PHISH: 1, Label.BENIGN: 0}.get(record.label, -1)
        return cls(
            page_id=record.page_id,
            url=record.host_url,
            html_path=Path(record.dom_html_path) if record.dom_html_path else None,
            screenshot_path=Path(record.screenshot_path) if record.screenshot_path else None,
            label=label,
            split=record.split or "",
            group_id=group_id,
        )

    def read_html(self) -> str | None:
        if self.html_path is None or not self.html_path.is_file():
            return None
        return self.html_path.read_text(encoding="utf-8", errors="replace")

    def has_screenshot(self) -> bool:
        return self.screenshot_path is not None and self.screenshot_path.is_file()


@dataclass
class Decision:
    """A detector's verdict. ``verdict`` is None when the page failed."""

    verdict: str | None
    brand: str | None
    basis: str
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class DetectorResult:
    verdict: str | None
    confidence: float | None
    brand: str | None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    name: str

    def extract(self, page: Page) -> Any: ...

    def decide(self, evidence: Any, page: Page, setting: dict[str, Any]) -> Decision: ...
