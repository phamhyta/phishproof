"""Hard cache for model calls.

Key = sha256(model + prompt + optional image hash). Every agent/baseline call goes
through this so (a) we never pay/recompute the same call twice, and (b) runs are
reproducible across the 5 seeds. Values are stored one-JSON-per-key on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, root: str | Path = "data/cache") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_image(data: bytes | str | Path) -> str:
        """Stable hash of image bytes (or a file path's contents)."""
        if isinstance(data, (str, Path)):
            data = Path(data).read_bytes()
        return hashlib.sha256(data).hexdigest()

    def _key(self, model: str, prompt: str, image_hash: str | None = None) -> str:
        h = hashlib.sha256()
        h.update(model.encode())
        h.update(b"\x00")
        h.update(prompt.encode())
        if image_hash:
            h.update(b"\x00")
            h.update(image_hash.encode())
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        # shard by first 2 hex chars to avoid huge flat dirs
        d = self.root / key[:2]
        d.mkdir(exist_ok=True)
        return d / f"{key}.json"

    def get(self, model: str, prompt: str, image_hash: str | None = None) -> Any | None:
        p = self._path(self._key(model, prompt, image_hash))
        if p.exists():
            return json.loads(p.read_text())["value"]
        return None

    def set(
        self, model: str, prompt: str, value: Any, image_hash: str | None = None
    ) -> None:
        key = self._key(model, prompt, image_hash)
        p = self._path(key)
        p.write_text(json.dumps({"model": model, "value": value}, ensure_ascii=False))
