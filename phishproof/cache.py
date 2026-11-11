"""On-disk cache of model responses, keyed by model, prompt and image hash."""

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
    def hash_image(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _key(model: str, prompt: str, image_hash: str | None) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode())
        digest.update(b"\x00")
        digest.update(prompt.encode())
        if image_hash:
            digest.update(b"\x00")
            digest.update(image_hash.encode())
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        shard = self.root / key[:2]
        shard.mkdir(exist_ok=True)
        return shard / f"{key}.json"

    def get(self, model: str, prompt: str, image_hash: str | None = None) -> Any | None:
        path = self._path(self._key(model, prompt, image_hash))
        return json.loads(path.read_text())["value"] if path.exists() else None

    def metadata(self, model: str, prompt: str, image_hash: str | None = None) -> dict:
        path = self._path(self._key(model, prompt, image_hash))
        return json.loads(path.read_text()).get("metadata", {}) if path.exists() else {}

    def set(
        self,
        model: str,
        prompt: str,
        value: Any,
        image_hash: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        path = self._path(self._key(model, prompt, image_hash))
        record = {"model": model, "value": value, "metadata": metadata or {}}
        path.write_text(json.dumps(record, ensure_ascii=False))
