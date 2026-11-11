"""Read and write page manifests (one JSON ``PageRecord`` per line)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

from ..schema import PageRecord


def slug(name: str) -> str:
    """A stable, filesystem-safe page id derived from a capture name."""
    base = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:60]
    return f"{base}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def write_manifest(records: Iterable[PageRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def read_manifest(path: str | Path) -> list[PageRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        return [PageRecord.model_validate_json(line) for line in handle if line.strip()]
