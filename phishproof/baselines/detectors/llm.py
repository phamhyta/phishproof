"""Language-model client with a response cache on disk.

Serves the LLM components of KnowPhish Sec. 5.2-5.3 and of PhishAgent Sec. 4.5 and App. C. Both papers use
GPT-3.5-turbo-instruct for text; this re-implementation uses GPT-4o for every component, at temperature 0 as
PhishAgent App. C states.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

DEFAULT_MODEL = "gpt-4o"
# image preprocessing matches the panel (RGB JPEG, quality 85, at most 768 px, detail low) because neither paper states one.
IMAGE_MAX_SIDE = 768
IMAGE_QUALITY = 85
IMAGE_DETAIL = "low"


class LLMError(RuntimeError):
    """A model request failed; the page is recorded as a failure and nothing is cached."""


class CacheMiss(LLMError):
    """A request was absent from the cache in cache-only mode."""


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    json_mode: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "json_mode": self.json_mode,
            "messages": self.messages,
        }

    def key(self) -> str:
        canonical = json.dumps(self.payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class LLM(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class OpenAIChat:
    """OpenAI chat completions client."""

    # the panel's client settings (120-second timeout and two retries) apply to the detectors too.
    def __init__(self, timeout: float = 120.0, max_retries: int = 2) -> None:
        from openai import OpenAI

        self._client = OpenAI(timeout=timeout, max_retries=max_retries)

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            result = self._client.chat.completions.create(
                model=request.model,
                temperature=request.temperature,
                messages=request.messages,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        choice = result.choices[0]
        usage = result.usage.model_dump() if result.usage is not None else {}
        return LLMResponse(
            text=choice.message.content or "",
            model=result.model,
            finish_reason=choice.finish_reason,
            usage=usage,
        )


def _redact_images(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace image bytes by their hash so that cache entries stay small and readable."""
    redacted = json.loads(json.dumps(payload))
    for message in redacted["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                part["image_url"]["url"] = "sha256:" + hashlib.sha256(url.encode("utf-8")).hexdigest()
    return redacted


class CachedLLM:
    """Wrap a client so that each distinct request is sent once and then served from disk."""

    def __init__(self, inner: LLM | None, cache_dir: str | Path, cache_only: bool = False) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_only = cache_only

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = request.key()
        path = self._path(key)
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=stored["text"],
                model=stored["model"],
                finish_reason=stored.get("finish_reason"),
                usage=stored.get("usage", {}),
                cached=True,
            )
        if self.cache_only or self.inner is None:
            raise CacheMiss(f"cache miss for request {key}")
        response = self.inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "key": key,
            "request": _redact_images(request.payload()),
            "text": response.text,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False)
        os.replace(temporary, path)
        return response


def system(text: str) -> dict[str, Any]:
    return {"role": "system", "content": text}


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def encode_image(path: str | Path) -> str:
    """Return the screenshot as a JPEG data URL, prepared as the panel's images are."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    rgb.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def user_with_image(text: str, image_path: str | Path) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": encode_image(image_path), "detail": IMAGE_DETAIL}},
        ],
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object in a reply, ignoring text around it; return None when there is none."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
