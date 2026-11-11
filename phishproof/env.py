"""Load API keys from a local .env file without overriding the environment."""

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
            env_file = base / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.split(" #")[0].strip().strip("\"'"))
                break
        _loaded = True
