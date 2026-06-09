"""Config loaders for the panel and the experiment.

The panel is defined in configs/panel.yaml so the model lineup (local Ollama models
+ optional GPT-4o spot-check) can change without touching code. All agents are
reached through an OpenAI-compatible endpoint (Ollama exposes one at /v1), so the
same client works for local and API models.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    id: str
    provider: str                 # "ollama" | "openai"
    model: str
    modality: str                 # "text" | "vision"
    base_url: str | None = None   # ollama: http://localhost:11434/v1 ; openai: None
    detail: str | None = None     # vision detail: "low" | "high" (OpenAI image cost lever)
    temperature: float = 0.0
    num_ctx: int = 8192           # Ollama context window (DOM + image can exceed 4096)
    timeout_s: float = 120.0      # per-call timeout so a hung request can't freeze the run


class SpotCheckConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o"
    n_pages: int = 150
    agreement_threshold: float = 0.80   # below this -> consider GPT-4o vision fallback


class PanelConfig(BaseModel):
    panel: list[AgentConfig]
    spot_check: SpotCheckConfig = Field(default_factory=SpotCheckConfig)


class ExperimentConfig(BaseModel):
    n_pages: int = 4000
    calibration_frac: float = 0.15
    seeds: list[int] = Field(default_factory=lambda: [0, 1])
    target_selective_risk: float = 0.01   # Cov99 operating point
    coverage_points: list[float] = Field(default_factory=lambda: [0.80])
    data_dir: str = "data/phishsel"
    cache_dir: str = "data/cache"
    results_dir: str = "results"


def load_panel(path: str | Path = "configs/panel.yaml") -> PanelConfig:
    return PanelConfig(**yaml.safe_load(Path(path).read_text()))


def load_experiment(path: str | Path = "configs/experiment.yaml") -> ExperimentConfig:
    return ExperimentConfig(**yaml.safe_load(Path(path).read_text()))
