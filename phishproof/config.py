"""Panel configuration, read from a YAML file so the model lineup can change without code edits."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class MemberConfig(BaseModel):
    id: str
    model: str
    modality: str  # "text" or "vision"
    base_url: str | None = None
    detail: str | None = None
    temperature: float = 0.0
    strict_schema: bool = False
    timeout_s: float = 120.0


class PanelConfig(BaseModel):
    panel: list[MemberConfig]


def load_panel(path: str | Path = "configs/panel.yaml") -> PanelConfig:
    return PanelConfig(**yaml.safe_load(Path(path).read_text()))
