"""Runtime configuration for the PhishProof pipeline.

The configuration schema is public (it documents the moving parts of the
system); the components it configures are released with the paper. Values here
are illustrative defaults, not the deployed operating point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class PanelConfig:
    """Which models make up the cross-modal panel.

    One vision agent reads the screenshot; the text agents read the page
    structure. The panel is deliberately small -- diversity, not count, drives
    the agreement signal.
    """

    vision_model: str = "<vision-model>"
    text_models: tuple[str, ...] = ("<text-model-a>", "<text-model-b>")
    max_tool_calls: int = 8


@dataclass
class ScoreConfig:
    """Aggregation and decision hyper-parameters.

    Attributes
    ----------
    epsilon:
        Weight of the text-agreement tie-breaker in the trust signal.
    tau:
        Calibrated act/abstain operating threshold on the trust score.
    active_cue_types:
        Evidence types the per-type agreement is averaged over.
    """

    epsilon: float = 1e-3
    tau: float = 0.5
    active_cue_types: tuple[str, ...] = (
        "brand",
        "form_action",
        "credential_intent",
        "logo",
    )


@dataclass
class ToolConfig:
    """Which grounding verifiers are enabled."""

    brand: bool = True
    dom: bool = True
    logo: bool = True
    certificate: bool = False
    redirect: bool = False


@dataclass
class PhishProofConfig:
    """Top-level configuration object."""

    panel: PanelConfig = field(default_factory=PanelConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    cache_dir: str = ".cache"

    @classmethod
    def from_json(cls, path: str | Path | None) -> "PhishProofConfig":
        """Load a configuration from a JSON file (or return defaults)."""
        if path is None:
            return cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            panel=PanelConfig(**raw.get("panel", {})),
            score=ScoreConfig(**raw.get("score", {})),
            tools=ToolConfig(**raw.get("tools", {})),
            cache_dir=raw.get("cache_dir", ".cache"),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
