from __future__ import annotations

import json
from pathlib import Path

from .detectors import Detector, UrlSignalDetector
from .io import read_pages, write_predictions
from .types import PageRecord, Prediction


class PhishProofRunner:
    def __init__(self, detector: Detector | None = None) -> None:
        self.detector = detector or UrlSignalDetector()

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "PhishProofRunner":
        config = _load_config(path)
        threshold = float(config.get("threshold", 0.55))
        return cls(UrlSignalDetector(threshold=threshold))

    def predict(self, page: PageRecord) -> Prediction:
        return self.detector.predict(page)

    def run_file(self, input_path: str | Path, output_path: str | Path) -> int:
        predictions = [self.predict(page) for page in read_pages(input_path)]
        write_predictions(output_path, predictions)
        return len(predictions)


def _load_config(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
