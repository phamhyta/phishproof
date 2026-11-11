"""OpenAI-compatible chat client with response caching and optional image input."""

from __future__ import annotations

import base64
import io
import os
import threading
from dataclasses import asdict, dataclass

from ..cache import JsonCache
from ..config import MemberConfig
from ..env import load_env
from .prompts import STRICT_RESPONSE_FORMAT

_VISION_MAX_PX = 768


@dataclass(frozen=True)
class ReplyDetails:
    """What the endpoint reported about a response, kept with the cached text."""

    served_model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class ModelReply:
    text: str
    details: ReplyDetails = ReplyDetails()


def _data_url(image_path: str, max_px: int = _VISION_MAX_PX) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > max_px:
        scale = max_px / max(width, height)
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _api_key(base_url: str | None) -> str:
    if base_url and "openrouter.ai" in base_url:
        return os.environ.get("OPENROUTER_API_KEY", "")
    return os.environ.get("OPENAI_API_KEY", "")


class ChatClient:
    def __init__(self, cache: JsonCache | None = None) -> None:
        self.cache = cache or JsonCache()
        self._clients: dict[str, object] = {}
        self._lock = threading.Lock()
        load_env()

    def _client(self, config: MemberConfig):
        key = config.base_url or "default"
        with self._lock:
            if key not in self._clients:
                from openai import OpenAI

                self._clients[key] = OpenAI(
                    base_url=config.base_url,
                    api_key=_api_key(config.base_url),
                    timeout=config.timeout_s,
                    max_retries=2,
                )
            return self._clients[key]

    def complete_json(
        self, config: MemberConfig, system: str, user: str, image_path: str | None = None
    ) -> str:
        return self.complete(config, system, user, image_path).text

    def complete(
        self, config: MemberConfig, system: str, user: str, image_path: str | None = None
    ) -> ModelReply:
        """Return the model's reply, from the cache when this exact request was made before."""
        image_hash = JsonCache.hash_image(image_path) if image_path else None
        cache_prompt = f"SYS:{system}\nUSR:{user}\nDETAIL:{config.detail}"
        if config.strict_schema:
            cache_prompt += "\nSTRICT"
        cached = self.cache.get(config.model, cache_prompt, image_hash)
        if cached is not None:
            details = self.cache.metadata(config.model, cache_prompt, image_hash)
            return ModelReply(cached, ReplyDetails(**details))

        response = self._client(config).chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": self._content(config, user, image_path)},
            ],
            temperature=config.temperature,
            response_format=(
                STRICT_RESPONSE_FORMAT if config.strict_schema else {"type": "json_object"}
            ),
        )
        choice = response.choices[0]
        text = choice.message.content or "{}"
        details = ReplyDetails(
            served_model=response.model,
            provider=getattr(response, "provider", None),
            finish_reason=choice.finish_reason,
            refusal=getattr(choice.message, "refusal", None),
        )
        self.cache.set(config.model, cache_prompt, text, image_hash, asdict(details))
        return ModelReply(text, details)

    @staticmethod
    def _content(config: MemberConfig, user: str, image_path: str | None) -> str | list[dict]:
        if not image_path or config.modality != "vision":
            return user
        image = {"url": _data_url(image_path)}
        if config.detail:
            image["detail"] = config.detail
        return [{"type": "text", "text": user}, {"type": "image_url", "image_url": image}]
