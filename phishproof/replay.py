"""Apply a frozen policy to stored per-page evidence, with no model calls."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .policy import Decision, FrozenPolicy, decide
from .schema import PageEvidence


def read_evidence(path: str | Path) -> Iterator[PageEvidence]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield PageEvidence.model_validate_json(line)


def write_evidence(evidence: Iterable[PageEvidence], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for page in evidence:
            handle.write(page.model_dump_json() + "\n")


def apply_policy(evidence: Iterable[PageEvidence], policy: FrozenPolicy) -> list[Decision]:
    return [decide(page, policy) for page in evidence]
