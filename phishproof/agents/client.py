"""LLM client with endpoint routing."""

from __future__ import annotations

from dataclasses import dataclass

from .._release import pending
from ..env import require_key


@dataclass
class LLMClient:
    """A thin client over a chat/vision model endpoint.

    The base URL selects the credentials (OpenAI vs OpenRouter). Requests are
    content-addressed by :mod:`phishproof.cache` so re-runs re-score for free.
    The request/response body is released with the paper.
    """

    model: str
    base_url: str | None = None
    temperature: float = 0.0

    def api_key(self) -> str:
        """Resolve the API key for this client's endpoint."""
        return require_key(self.base_url)

    def complete(self, prompt: str, image: str | None = None) -> str:
        """Return the model's raw response for a prompt (and optional image)."""
        pending("agents.client.LLMClient.complete")
