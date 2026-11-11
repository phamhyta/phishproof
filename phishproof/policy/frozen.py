"""The policy settings fixed on calibration data before any test use."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from ..calibration import IsotonicCalibrator
from ..schema import PageEvidence
from .thresholds import scan_block, scan_clear

DEFAULT_LOGO_THRESHOLD = 0.96


class FrozenPolicy(BaseModel):
    """Thresholds on the ranking score. A threshold of None disables that action."""

    tau_block: float | None
    tau_clear: float | None
    alpha_block: float = 0.02
    alpha_clear: float = 0.02
    delta: float = 0.05
    logo_threshold: float = DEFAULT_LOGO_THRESHOLD
    calibrator: dict | None = None

    def correctness(self, rank: float | None) -> float | None:
        """Calibrated correctness estimate reported with an action. It selects no threshold."""
        if rank is None or self.calibrator is None:
            return None
        return IsotonicCalibrator.from_dict(self.calibrator)(rank)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> FrozenPolicy:
        return cls.model_validate_json(Path(path).read_text())


def fit_policy(
    evidence: Sequence[PageEvidence],
    groups: Mapping[str, str],
    alpha_block: float = 0.02,
    alpha_clear: float = 0.02,
    delta: float = 0.05,
    logo_threshold: float = DEFAULT_LOGO_THRESHOLD,
) -> FrozenPolicy:
    """Fit on a labelled calibration split that shares no duplicate group with any test split."""
    ranked = [page for page in evidence if page.complete and page.rank is not None]
    calibrator = IsotonicCalibrator().fit(
        [page.rank for page in ranked], [page.verdict is page.label for page in ranked]
    )
    return FrozenPolicy(
        tau_block=scan_block(evidence, groups, alpha_block, delta).level,
        tau_clear=scan_clear(evidence, groups, alpha_clear, delta).level,
        alpha_block=alpha_block,
        alpha_clear=alpha_clear,
        delta=delta,
        logo_threshold=logo_threshold,
        calibrator=calibrator.to_dict(),
    )
