"""Follow-up confidence request, sent after a detector has decided so the decision cannot change."""

from __future__ import annotations

import math
from typing import Any

from .base import Page
from .llm import (
    DEFAULT_MODEL,
    LLM,
    LLMError,
    LLMRequest,
    parse_json_object,
    system,
    user,
    user_with_image,
)
from .pagetext import plain_text

# this wording approximates a panel member's request, because the panel's prompt is outside this folder; align it with that prompt before running.
SYSTEM_PROMPT = "You are a phishing analyst. Reply with one JSON object and nothing else."
USER_PROMPT = """A phishing detector has already decided about the webpage below. The decision is final and your answer will not change it.

URL: {url}
Page text: {text}
Decision: {verdict}
Brand: {brand}
Reason: {basis}

How confident are you that this decision is correct? Return a JSON object with one key, "confidence", whose value is a number from 0 to 1."""
# the follow-up sees 4,000 characters of page text, because no source fixes a length.
TEXT_LIMIT = 4_000


def parse_confidence(text: str) -> float | None:
    """Return a finite confidence in [0, 1]; a missing or invalid value stays missing and is never imputed."""
    parsed = parse_json_object(text)
    if parsed is None:
        return None
    value = parsed.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def ask_confidence(
    llm: LLM,
    page: Page,
    verdict: str,
    brand: str | None,
    basis: str,
    use_screenshot: bool = False,
    model: str = DEFAULT_MODEL,
) -> tuple[float | None, dict[str, Any]]:
    """Request a confidence for a decision that has already been made."""
    html = page.read_html() or ""
    prompt = USER_PROMPT.format(
        url=page.url,
        text=plain_text(html, TEXT_LIMIT),
        verdict=verdict,
        brand=brand or "none identified",
        basis=basis,
    )
    # the screenshot is attached only when a vision component took part in the decision, so a text-only decision stays text-only.
    message = user_with_image(prompt, page.screenshot_path) if use_screenshot and page.has_screenshot() else user(prompt)
    request = LLMRequest(messages=[system(SYSTEM_PROMPT), message], model=model, json_mode=True)
    try:
        response = llm.complete(request)
    except LLMError as exc:
        return None, {"error": str(exc)}
    return parse_confidence(response.text), {"text": response.text, "model": response.model}
