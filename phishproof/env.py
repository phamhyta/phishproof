"""Tiny .env loader (no extra dependency).

Reads KEY=VALUE lines from a .env file in the project root and populates os.environ
without overriding already-set variables. Lets you keep OPENAI_API_KEY in a gitignored
.env file instead of exporting it in the shell.
"""

from __future__ import annotations

import os
from pathlib import Path

_loaded = False


def load_env(start: Path | None = None) -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
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
            return
