"""Thin OpenAI-compatible chat client with hard caching and optional image input.

One code path serves both local Ollama models (base_url=http://localhost:11434/v1)
and GPT-4o spot-check (base_url=None, real api_key), because Ollama exposes an
OpenAI-compatible API. Every call is cached on (model, prompt, image hash).
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from ..cache import JsonCache
from ..config import AgentConfig

# Downscale screenshots before sending to the VLM: a 1920x1080 PNG is very slow to encode
# on a CPU VLM. 768px on the long side keeps logo/layout legible at a fraction of the cost.
_VISION_MAX_PX = 768


def _data_url(image_path: str, max_px: int = _VISION_MAX_PX) -> str:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class ChatClient:
    def __init__(self, cache: JsonCache | None = None) -> None:
        self.cache = cache or JsonCache()
        self._clients: dict[str, object] = {}

    def _client(self, cfg: AgentConfig):
        key = cfg.base_url or "openai-default"
        if key not in self._clients:
            from openai import OpenAI

            from ..env import load_env

            load_env()  # pick up OPENAI_API_KEY from a .env file if present
            api_key = os.environ.get("OPENAI_API_KEY", "ollama" if cfg.base_url else "")
            # Per-call timeout + bounded retries so one hung request can't freeze the run.
            self._clients[key] = OpenAI(
                base_url=cfg.base_url,
                api_key=api_key or "ollama",
                timeout=cfg.timeout_s,
                max_retries=2,
            )
        return self._clients[key]

    def complete_json(
        self, cfg: AgentConfig, system: str, user: str, image_path: str | None = None
    ) -> str:
        """Return the model's raw JSON string content (cached)."""
        img_hash = JsonCache.hash_image(image_path) if image_path else None
        cache_prompt = f"SYS:{system}\nUSR:{user}\nDETAIL:{cfg.detail}"
        cached = self.cache.get(cfg.model, cache_prompt, img_hash)
        if cached is not None:
            return cached

        content: list[dict] | str
        if image_path and cfg.modality == "vision":
            img_block = {"type": "image_url",
                         "image_url": {"url": _data_url(image_path)}}
            if cfg.detail:
                img_block["image_url"]["detail"] = cfg.detail
            content = [{"type": "text", "text": user}, img_block]
        else:
            content = user

        client = self._client(cfg)
        extra: dict = {}
        if cfg.base_url:  # Ollama: raise context window via options
            extra["extra_body"] = {"options": {"num_ctx": cfg.num_ctx}}
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=cfg.temperature,
            response_format={"type": "json_object"},
            **extra,
        )
        out = resp.choices[0].message.content or "{}"
        self.cache.set(cfg.model, cache_prompt, out, img_hash)
        return out
