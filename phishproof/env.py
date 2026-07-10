"""Environment and credential resolution.

Resolves the API keys the panel clients need. Reading the environment is public;
the clients that consume the keys are released with the paper.
"""

from __future__ import annotations

import os

OPENAI_KEY_VAR = "OPENAI_API_KEY"
OPENROUTER_KEY_VAR = "OPENROUTER_API_KEY"


def resolve_key(base_url: str | None) -> str | None:
    """Pick the API key that matches a client's base URL.

    OpenRouter endpoints use ``OPENROUTER_API_KEY``; everything else falls back
    to ``OPENAI_API_KEY``. Returns ``None`` when the variable is unset so the
    caller can raise a helpful error.
    """
    if base_url and "openrouter" in base_url:
        return os.environ.get(OPENROUTER_KEY_VAR)
    return os.environ.get(OPENAI_KEY_VAR)


def require_key(base_url: str | None) -> str:
    """Like :func:`resolve_key` but raise when the key is missing."""
    key = resolve_key(base_url)
    if not key:
        var = OPENROUTER_KEY_VAR if (base_url and "openrouter" in base_url) else OPENAI_KEY_VAR
        raise RuntimeError(f"missing API key: set {var} in the environment")
    return key
