"""End-to-end PhishProof pipeline.

The runner wires together the five stages of the method:

1. Build a cross-modal context for the page (screenshot + DOM + URL).
2. Run the agent panel to gather typed, cited evidence cues.
3. Score per-type agreement (GEA) and ground each agreed cue with a verifier.
4. Map the trust signal to a calibrated correctness probability.
5. Act above the operating threshold, else abstain; return the grounded cues as
   the checkable explanation.

The orchestration surface is public; each stage's implementation is released
with the paper.
"""

from __future__ import annotations

from pathlib import Path

from ._release import pending
from .agents import AgentPanel
from .calibration import IsotonicCalibrator
from .config import PhishProofConfig
from .schema import EvidenceBundle
from .tools import ToolRegistry, default_registry
from .types import PageRecord, Prediction


class PhishProofRunner:
    """Selective phishing detector over a cross-modal agent panel."""

    def __init__(
        self,
        panel: AgentPanel | None = None,
        tools: ToolRegistry | None = None,
        calibrator: IsotonicCalibrator | None = None,
        config: PhishProofConfig | None = None,
    ) -> None:
        self.config = config or PhishProofConfig()
        self.panel = panel
        self.tools = tools or default_registry()
        self.calibrator = calibrator

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "PhishProofRunner":
        """Build a runner (panel + tools + calibrator) from a config file."""
        pending("pipeline.PhishProofRunner.from_config")

    def gather(self, page: PageRecord) -> EvidenceBundle:
        """Stages 1-3: build context, run the panel, score agreement, ground cues."""
        pending("pipeline.PhishProofRunner.gather")

    def predict(self, page: PageRecord) -> Prediction:
        """Full pipeline for one page: gather -> calibrate -> act/abstain."""
        pending("pipeline.PhishProofRunner.predict")

    def run_file(self, input_path: str | Path, output_path: str | Path) -> int:
        """Score every page in a JSONL file and write the predictions."""
        pending("pipeline.PhishProofRunner.run_file")
