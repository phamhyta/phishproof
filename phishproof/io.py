from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .types import PageRecord, Prediction


def read_pages(path: str | Path) -> Iterator[PageRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                yield PageRecord.from_json(row)
            except Exception as exc:
                raise ValueError(f"invalid JSONL record at line {line_no}") from exc


def write_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_json(), sort_keys=True) + "\n")
