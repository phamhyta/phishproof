"""Content-addressed cache for model and tool calls.

Every panel/tool call is keyed by a hash of its inputs so a re-run re-scores for
free instead of re-calling the models. The key scheme is public; the code that
populates the cache (the agent and tool calls) is released with the paper.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(*, model: str, prompt: str, image: str | None = None) -> str:
    """Deterministic key for a single call.

    The key is ``sha256(model || prompt || image-bytes-or-empty)`` so identical
    inputs always map to the same entry regardless of endpoint.
    """
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(prompt.encode("utf-8"))
    hasher.update(b"\x00")
    if image:
        hasher.update(Path(image).read_bytes() if Path(image).exists() else image.encode())
    return hasher.hexdigest()


class JsonCache:
    """A tiny JSON-file cache under ``root``.

    The lookup/scan plumbing is real; the values it stores (model responses and
    tool outputs) are produced by the withheld implementation.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def put(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
