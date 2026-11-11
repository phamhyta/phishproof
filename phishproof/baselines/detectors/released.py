"""Adapters around the released code of Phishpedia, PhishIntention and PhishLLM.

Each released package must be installed separately, with its own models. The captures are
archived, so no adapter opens a live page.
"""

from __future__ import annotations

from typing import Any

from .base import BENIGN, PHISH, DetectorResult, Page
from .confidence import ask_confidence
from .llm import DEFAULT_MODEL, LLM


def _paths(page: Page) -> tuple[str, str]:
    return str(page.screenshot_path), str(page.html_path)


class PhishpediaDetector:
    """D1. The confidence is the logo-match similarity."""

    name = "D1-Phishpedia"

    def __init__(self, wrapper: Any = None, logo_match: float = 0.83) -> None:
        if wrapper is None:
            from phishpedia import PhishpediaWrapper

            wrapper = PhishpediaWrapper()
        wrapper.SIAMESE_THRE = logo_match
        self.wrapper = wrapper

    def predict(self, page: Page) -> DetectorResult:
        screenshot, html = _paths(page)
        category, target, _, _, similarity, *_ = self.wrapper.test_orig_phishpedia(
            page.url, screenshot, html
        )
        return DetectorResult(PHISH if category == 1 else BENIGN, similarity, target)


class PhishIntentionDetector:
    """D2. Runs the layout, logo and credential-page steps, and skips the live dynamic step."""

    name = "D2-PhishIntention"
    _NOT_CREDENTIAL_PAGE = 1

    def __init__(self, wrapper: Any = None, logo_match: float = 0.80) -> None:
        if wrapper is None:
            from phishintention.pipeline import PhishIntentionWrapper

            wrapper = PhishIntentionWrapper()
        wrapper.SIAMESE_THRE = logo_match
        self.wrapper = wrapper

    @staticmethod
    def _logo_boxes(boxes: Any, classes: Any) -> Any:
        from phishintention.modules.awl_detector import find_element_type

        return find_element_type(boxes, classes, bbox_type="logo")[0]

    def predict(self, page: Page) -> DetectorResult:
        screenshot, html = _paths(page)
        boxes, classes, _, _ = self.wrapper._step1_layout_detector(screenshot)
        if boxes is None or len(boxes) == 0:
            return DetectorResult(BENIGN, None, None)
        logo_boxes = self._logo_boxes(boxes, classes)
        if logo_boxes is None or len(logo_boxes) == 0:
            return DetectorResult(BENIGN, None, None)
        target, _, _, similarity, _ = self.wrapper._step2_logo_matcher(
            logo_boxes, page.url, screenshot
        )
        if target is None:
            return DetectorResult(BENIGN, similarity, None)
        credential_class, _ = self.wrapper._step3_crp_classifier(screenshot, html, boxes, classes)
        verdict = BENIGN if credential_class == self._NOT_CREDENTIAL_PAGE else PHISH
        return DetectorResult(verdict, similarity, target)


class PhishLLMDetector:
    """D3, without the interactive login search. The confidence comes from a follow-up request."""

    name = "D3-PhishLLM"

    def __init__(self, vlm: Any, llm: LLM, model: str = DEFAULT_MODEL) -> None:
        self.vlm = vlm
        self.llm = llm
        self.model = model

    @classmethod
    def from_release(cls, param_dict: dict, llm: LLM, model: str = DEFAULT_MODEL) -> PhishLLMDetector:
        from scripts.phishintention.model_config import load_config
        from scripts.pipeline.phishvlm import PhishVLM
        from scripts.utils.PhishIntentionWrapper import LayoutDetector, LogoDetector, LogoEncoder

        awl_model, siamese_model, ocr_model, siamese_threshold = load_config()
        vlm = PhishVLM(
            logo_encoder=LogoEncoder(siamese_model, ocr_model, siamese_threshold),
            logo_extractor=LogoDetector(awl_model),
            layout_extractor=LayoutDetector(awl_model),
            param_dict=param_dict,
        )
        return cls(vlm, llm, model)

    def predict(self, page: Page) -> DetectorResult:
        screenshot, html = _paths(page)
        logo_box, logo = self.vlm.detect_logo(screenshot)
        # At the interaction limit the release reports the page without clicking through it.
        limit = getattr(self.vlm, "interaction_limit", 0)
        verdict, target, *_ = self.vlm.test(
            page.url, logo, logo_box, screenshot, html, driver=None, limit=limit
        )
        brand = None if target in (None, "None") else target
        confidence, response = ask_confidence(
            self.llm, page, verdict, brand, "PhishLLM decision", use_screenshot=True, model=self.model
        )
        return DetectorResult(verdict, confidence, brand, raw={"confidence_response": response})
