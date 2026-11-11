"""Run a re-implemented detector on one page and attach its follow-up confidence."""

from __future__ import annotations

import logging
from typing import Any

from .base import Decision, Detector, DetectorResult, Page
from .confidence import ask_confidence
from .llm import DEFAULT_MODEL, LLM

log = logging.getLogger(__name__)


def _decide(detector: Detector, page: Page, setting: dict[str, Any]) -> Decision:
    try:
        return detector.decide(detector.extract(page), page, setting)
    except Exception as error:  # one bad page must not stop the others
        log.exception("%s failed on page %s", detector.name, page.page_id)
        return Decision(None, None, "page failed", error=f"{type(error).__name__}: {error}")


def predict_page(
    detector: Detector, llm: LLM, page: Page, setting: dict[str, Any], model: str = DEFAULT_MODEL
) -> DetectorResult:
    decision = _decide(detector, page, setting)
    confidence, response = None, None
    if decision.verdict is not None:
        confidence, response = ask_confidence(
            llm,
            page,
            decision.verdict,
            decision.brand,
            decision.basis,
            use_screenshot=bool(decision.raw.get("vision_used")),
            model=model,
        )
    return DetectorResult(
        verdict=decision.verdict,
        confidence=confidence,
        brand=decision.brand,
        error=decision.error,
        raw={**decision.raw, "basis": decision.basis, "confidence_response": response},
    )
