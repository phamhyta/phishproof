"""Tiny .env loader (no extra dependency).

Reads KEY=VALUE lines from a .env file in the project root and populates os.environ
without overriding already-set variables. Lets you keep OPENAI_API_KEY in a gitignored
.env file instead of exporting it in the shell.

Thread-safe + idempotent: the whole load runs under a lock and `_loaded` flips to True
only AFTER the .env is applied, so a concurrent caller can never observe `_loaded=True`
while os.environ is still unpopulated (that race made parallel runs build an OpenAI client
with an empty key -> api_key falls back to "ollama" -> 401 against api.openai.com).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

_loaded = False
_lock = threading.Lock()


def load_env(start: Path | None = None) -> None:
    global _loaded
    with _lock:
        if _loaded:
            return
        here = (start or Path(__file__)).resolve()
        for base in [Path.cwd(), *here.parents]:
            env = base / ".env"
            if env.exists():
                for line in env.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
                break
        _loaded = True  # only after .env is fully applied (race-safe)
