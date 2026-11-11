"""PhishAgent in its offline-only configuration (comparator D5).

Implements PhishAgent Sec. 4.2 (preprocessing), Sec. 4.3 (offline knowledge-based module) and Sec. 4.5 (Agent Core:
Multimodal Brand Extractor, Domain Checker, Target Brand Checker, Recheck Procedure), with the prompts of App. C. The
online module of Sec. 4.4 is removed, so Eq. (2) uses only the offline sets.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image

from .base import BENIGN, PHISH, Decision, Page
from .domains import registrable_domain
from .kb import Brand, KnowledgeBase
from .knowphish import NO_BRAND
from .llm import DEFAULT_MODEL, LLM, LLMError, LLMRequest, system, user, user_with_image
from .logos import LogoBackend
from .pagetext import plain_text
from .retriever import Hit

MAX_K = 7
# at most 20 aliases per brand are shown to the language models, to bound tokens, because the paper does not cap them.
PROMPT_ALIASES = 20

SYSTEM_PROMPT = "You are a helpful assistant that responds in detecting brand name in a webpage."
TBE_PROMPT = """Define targeted brand as a brand that a webpage belongs to. Given the URL and HTML along with the potential brands of a webpage P, answer:
(1) What the targeted brand of P is. The target brand can appear in the HTML or the URL, please look carefully. There are cases where the target brand appears in the url along with other characters, try to identify the target brand from the URL. If the analysis suggests that the brand name is not among the provided list of potential brands, independently identify and determine the correct brand name based on the content. Extract the brand name only and do not include extra details such as affiliated products, countries, or additional abbreviations; If you are not sure about what the targeted brand of P is, please output a brand name from the list of potential brands that you think it is the most relevant to the webpage P. If the brand is not identifiable from the given information, put "Not identifiable". You only output the brand name without any additional explanation.
Input information:
-URL:
"{url}"
-HTML:
"{processed_html}"
Potential brands:
"{potential_brands}"
Answer:"""
IBE_PROMPT = """Analyze the provided image to detect a single, authentic brand name, ensuring its accuracy even in the presence of adversarial attacks. If no brand name is identifiable in the image, return "Not identifiable". Output only the genuine brand name without additional explanation. If there are multiple brand names in the image, identify the most suitable and authentic brand name associated with the genuine logo."""
RECHECK_PROMPT = """(1) Analyze the provided image to detect a single, authentic brand name, ensuring its accuracy even in the presence of adversarial attacks. If no brand name is identifiable in the image, return "Not identifiable". Output only the genuine brand name without additional explanation. If there are multiple brand names in the image, identify the most suitable and authentic brand name associated with the genuine logo.
(2) Does the detected brand name in (1) match the brand "{old_brand}"? If yes, return "Yes"; if no, return "No" without any explanation.
Example output format:
(1) Singtel
(2) Yes"""
# The Google search list of the published prompt is removed because this configuration is offline-only.
TBC_PROMPT = """Given a list of items found from Knowledge Base.
- The full list of items found from Knowledge Base is as follows:
"{offline_items}".
Based on the given searched items, determine if any of the provided searched items mention to the brand "{brand}". If yes, return 1; if not, return 0.
Example output: 1"""


class Retriever(Protocol):
    def search(self, text: str, logo: Image.Image | None, limit: int) -> list[Hit]: ...


def clean_brand(text: str) -> str:
    """Reduce a free-text reply to one brand name."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0].strip().strip("\"'`.").strip()


def is_identified(brand: str) -> bool:
    return bool(brand) and brand.casefold() not in NO_BRAND and "not identifiable" not in brand.casefold()


def parse_recheck(text: str) -> tuple[str, bool] | None:
    """Parse "(1) brand / (2) Yes|No"; return None when the format is not recognisable."""
    first = re.search(r"\(1\)\s*(.+)", text)
    second = re.search(r"\(2\)\s*(yes|no)\b", text, flags=re.IGNORECASE)
    if first is None or second is None:
        return None
    return clean_brand(first.group(1)), second.group(1).lower() == "yes"


def parse_tbc(text: str) -> bool | None:
    match = re.search(r"\b([01])\b", text)
    return None if match is None else match.group(1) == "1"


def _with_aliases(brand: Brand) -> str:
    others = [alias for alias in brand.aliases if alias != brand.name][:PROMPT_ALIASES]
    return f"{brand.name}, also known as {', '.join(others)}" if others else brand.name


@dataclass
class AgentEvidence:
    domain: str
    url: str
    text: str
    hits: list[Hit]
    logo_found: bool
    screenshot: bool
    error: str | None = None


class PhishAgent:
    """Agent Core of PhishAgent over the offline knowledge-based module."""

    name = "d5"

    def __init__(
        self, llm: LLM, kb: KnowledgeBase, retriever: Retriever, logos: LogoBackend, model: str = DEFAULT_MODEL
    ) -> None:
        self.llm = llm
        self.kb = kb
        self.retriever = retriever
        self.logos = logos
        self.model = model

    def extract(self, page: Page) -> AgentEvidence:
        """Preprocess a page and retrieve up to MAX_K brands; k and the floor are applied in decide()."""
        html = page.read_html()
        if html is None:
            # a page without HTML is a failure, not a benign verdict, matching the paper's treatment of missing outputs.
            return AgentEvidence(registrable_domain(page.url), page.url, "", [], False, False, "missing_html")
        text = plain_text(html)
        logo = self.logos.top_logo(page.screenshot_path) if page.has_screenshot() else None
        hits = self.retriever.search(text, logo, MAX_K)
        return AgentEvidence(registrable_domain(page.url), page.url, text, hits, logo is not None, page.has_screenshot())

    def _ask(self, messages: list[dict[str, Any]]) -> str:
        return self.llm.complete(LLMRequest(messages=messages, model=self.model)).text

    def _brand_from_text(self, evidence: AgentEvidence, brands: Sequence[Brand]) -> tuple[str, str]:
        potential = "; ".join(_with_aliases(brand) for brand in brands)
        prompt = TBE_PROMPT.format(url=evidence.url, processed_html=evidence.text, potential_brands=potential)
        output = self._ask([system(SYSTEM_PROMPT), user(prompt)])
        return clean_brand(output), output

    def _brand_from_image(self, page: Page) -> tuple[str, str]:
        output = self._ask([system(SYSTEM_PROMPT), user_with_image(IBE_PROMPT, page.screenshot_path)])
        return clean_brand(output), output

    def _recheck(self, page: Page, old_brand: str) -> tuple[tuple[str, bool] | None, str]:
        prompt = RECHECK_PROMPT.format(old_brand=old_brand)
        output = self._ask([system(SYSTEM_PROMPT), user_with_image(prompt, page.screenshot_path)])
        return parse_recheck(output), output

    def _in_knowledge(self, target: str, brands: Sequence[Brand], raw: dict[str, Any], slot: str) -> bool:
        """Run the Target Brand Checker over the retrieved brands."""
        # each item is listed as name, aliases and domains, because App. C does not fix the format of the offline items.
        items = "; ".join(f"{_with_aliases(brand)}; domains: {', '.join(brand.domains)}" for brand in brands)
        prompt = TBC_PROMPT.format(offline_items=items, brand=target)
        output = self._ask([user(prompt)])
        raw[slot] = output
        parsed = parse_tbc(output)
        # an unparseable Target Brand Checker reply counts as "not found", which is the benign path.
        return bool(parsed)

    def decide(self, evidence: AgentEvidence, page: Page, setting: dict[str, Any]) -> Decision:
        k, floor = int(setting["k"]), float(setting["floor"])
        raw: dict[str, Any] = {"domain": evidence.domain, "logo_found": evidence.logo_found, "vision_used": False}
        if evidence.error is not None:
            return Decision(None, None, "page failed", raw, evidence.error)
        # the similarity floor drops retrieved brands before the Agent Core sees them; PhishAgent has no floor, and the paper's tuning grid adds one.
        hits = [hit for hit in evidence.hits if hit.score >= floor][:k]
        brands = [self.kb[hit.brand_id] for hit in hits]
        raw["retrieved"] = [{"brand": self.kb[hit.brand_id].name, "score": hit.score} for hit in hits]
        domains = {domain for brand in brands for domain in brand.domains}
        try:
            return self._agent_core(evidence, page, brands, domains, raw)
        except LLMError as exc:
            return Decision(None, None, "page failed", raw, f"llm: {exc}")

    def _agent_core(
        self, evidence: AgentEvidence, page: Page, brands: list[Brand], domains: set[str], raw: dict[str, Any]
    ) -> Decision:
        target, raw["tbe_response"] = self._brand_from_text(evidence, brands)
        used_image_extractor = False
        if not is_identified(target):
            if not evidence.screenshot:
                raw["target_brand"] = None
                return Decision(BENIGN, None, "No target brand could be identified from the text and no screenshot exists.", raw)
            target, raw["ibe_response"] = self._brand_from_image(page)
            used_image_extractor, raw["vision_used"] = True, True
            if not is_identified(target):
                raw["target_brand"] = None
                return Decision(BENIGN, None, "No target brand could be identified from the text or the screenshot.", raw)
        raw["target_brand"] = target

        if evidence.domain in domains:
            return Decision(BENIGN, target, "The page's domain is among the domains of the retrieved brands.", raw)
        # with no retrieved brand the Target Brand Checker and Recheck are skipped, because an empty offline set cannot make Eq. (2) true.
        if not brands:
            return Decision(BENIGN, target, "No brand passed retrieval, so the target brand is not in the knowledge base.", raw)
        if self._in_knowledge(target, brands, raw, "tbc_response"):
            return Decision(PHISH, target, f"The target brand {target} is in the knowledge base but the page's domain is not.", raw)
        if used_image_extractor or not evidence.screenshot:
            return Decision(BENIGN, target, "The target brand is not in the knowledge base.", raw)

        rechecked, raw["recheck_response"] = self._recheck(page, target)
        raw["vision_used"] = True
        if rechecked is None:
            # an unparseable Recheck reply is treated as same_old=True, which is the benign path.
            return Decision(BENIGN, target, "The screenshot recheck reply could not be parsed.", raw)
        new_target, same_old = rechecked
        raw["recheck_brand"], raw["recheck_same"] = new_target, same_old
        if same_old or not is_identified(new_target):
            return Decision(BENIGN, target, "The screenshot recheck found no different target brand.", raw)
        raw["target_brand"] = new_target
        # The offline sets do not change after Recheck, so the Domain Checker's negative answer still holds.
        if self._in_knowledge(new_target, brands, raw, "tbc_recheck_response"):
            return Decision(PHISH, new_target, f"The rechecked brand {new_target} is in the knowledge base but the page's domain is not.", raw)
        return Decision(BENIGN, new_target, "The rechecked brand is not in the knowledge base.", raw)
