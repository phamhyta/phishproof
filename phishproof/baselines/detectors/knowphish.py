"""KnowPhish Detector (KPD, comparator D4).

Implements KnowPhish Sec. 5 and Algorithm 1: LLM summary (Sec. 5.2), text-based CRP classification (Sec. 5.1), logo and
text brand extraction with exact alias matching (Sec. 5.3), and the domain check (Sec. 5.4).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .base import BENIGN, PHISH, Decision, Page
from .domains import registrable_domain
from .kb import KnowledgeBase
from .llm import DEFAULT_MODEL, LLM, LLMError, LLMRequest, parse_json_object, system, user
from .logos import LogoBackend
from .pagetext import kpd_html

NO_BRAND = {"", "none", "null", "n/a", "unknown", "not identifiable"}

# the summary prompt and its three in-context examples are written here, because Sec. 5.2 keeps the template in a supplementary repository that was not consulted.
SUMMARY_SYSTEM = "You analyse webpages for a phishing detector. Reply with one JSON object and nothing else."
SUMMARY_INSTRUCTION = """Given a webpage's URL and HTML, with scripts and styles removed, write a summary as a JSON object with these keys:
- "brand": the brand the page presents itself as belonging to, judged from its content and URL, as a plain name; null if no brand can be identified.
- "is_crp": true if the page requires credentials, either directly with input fields (explicit) or through an element such as a login button that leads to such a form (implicit); otherwise false.
- "crp_elements": the HTML elements that make it a credential-requiring page; an empty list if there are none.
- "rationale": one or two sentences explaining the "is_crp" value.

Example 1
URL: http://secure-login.example-host.net/signin
HTML: <title>Log in to your PayPal account</title><form><input type="email" name="login"><input type="password" name="pw"><button>Log In</button></form>
Summary: {"brand": "PayPal", "is_crp": true, "crp_elements": ["input type=password", "input type=email"], "rationale": "The page asks directly for an e-mail address and a password."}

Example 2
URL: http://parcel-update.example.org/home
HTML: <title>DHL Express</title><h1>Your parcel is waiting</h1><a href="/verify" class="btn">Sign in to reschedule delivery</a>
Summary: {"brand": "DHL", "is_crp": true, "crp_elements": ["a.btn Sign in to reschedule delivery"], "rationale": "The page shows no input fields but its sign-in button leads to a credential form, an implicit CRP."}

Example 3
URL: https://www.citynews.example/articles/weather
HTML: <title>Local weather</title><h1>Rain expected this weekend</h1><p>Forecasters expect showers on Saturday.</p>
Summary: {"brand": "City News", "is_crp": false, "crp_elements": [], "rationale": "The page is a news article and requests no credentials."}
"""
SUMMARY_TEMPLATE = "{instruction}\nNow summarise this page.\nURL: {url}\nHTML: {html}\nSummary:"


def has_brand(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() not in NO_BRAND


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no"}:
        return False
    return None


class CrpClassifier(Protocol):
    def predict(self, processed_html: str, summary: dict[str, Any]) -> bool | None: ...


class SummaryCrp:
    """Read the CRP decision from the GPT-4o summary."""

    # the default CRP decision is the summary's is_crp field, because the paper's XLM-RoBERTa was trained on 2,555 hand-labelled pages that are not available.
    def predict(self, processed_html: str, summary: dict[str, Any]) -> bool | None:
        return as_bool(summary.get("is_crp"))


class XlmRobertaCrp:
    """Sec. 5.1 classifier, loaded from a checkpoint trained elsewhere; it takes the HTML and the LLM summary."""

    def __init__(self, checkpoint: str | Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(self._device).eval()
        self._lock = threading.Lock()

    def predict(self, processed_html: str, summary: dict[str, Any]) -> bool | None:
        # the summary and HTML form a sequence pair truncated to 512 tokens, and class 1 means CRP, because Sec. 5.1 does not state how they are combined.
        inputs = self._tokenizer(
            json.dumps(summary, ensure_ascii=False), processed_html, truncation=True, max_length=512, return_tensors="pt"
        ).to(self._device)
        with self._lock, self._torch.no_grad():
            logits = self._model(**inputs).logits
        return int(logits.argmax(-1).item()) == 1


@dataclass
class KPDEvidence:
    """Everything KPD needs for a decision; tuning reuses it across logo thresholds at no cost."""

    domain: str
    error: str | None = None
    summary_text: str | None = None
    summary: dict[str, Any] | None = None
    model: str | None = None
    is_crp: bool | None = None
    logo_found: bool = False
    logo_brand_id: str | None = None
    logo_similarity: float | None = None
    text_brand: str | None = None
    text_brand_ids: list[str] = field(default_factory=list)


class KPD:
    """Algorithm 1 of KnowPhish over the rebuilt knowledge base."""

    name = "d4"

    def __init__(
        self,
        llm: LLM,
        kb: KnowledgeBase,
        logos: LogoBackend,
        crp: CrpClassifier | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.llm = llm
        self.kb = kb
        self.logos = logos
        self.crp = crp or SummaryCrp()
        self.model = model

    def extract(self, page: Page) -> KPDEvidence:
        evidence = KPDEvidence(domain=registrable_domain(page.url))
        html = page.read_html()
        if html is None:
            # a page without HTML is a failure, not a benign verdict, matching the paper's treatment of missing outputs.
            evidence.error = "missing_html"
            return evidence
        processed = kpd_html(html)
        # the summary receives the full URL as Sec. 5.2 says, although Algorithm 1 line 1 writes w.domain.
        prompt = SUMMARY_TEMPLATE.format(instruction=SUMMARY_INSTRUCTION, url=page.url, html=processed)
        request = LLMRequest(messages=[system(SUMMARY_SYSTEM), user(prompt)], model=self.model, json_mode=True)
        try:
            response = self.llm.complete(request)
        except LLMError as exc:
            evidence.error = f"llm: {exc}"
            return evidence
        evidence.summary_text, evidence.model = response.text, response.model
        evidence.summary = parse_json_object(response.text)
        if evidence.summary is None:
            evidence.error = "summary_unparseable"
            return evidence
        evidence.is_crp = self.crp.predict(processed, evidence.summary)
        if evidence.is_crp is None:
            evidence.error = "crp_unparseable"
            return evidence
        if page.has_screenshot():
            # only the highest-scoring detected logo is matched, as PhishAgent App. C does, because KnowPhish does not say how many boxes are tried.
            logo = self.logos.top_logo(page.screenshot_path)
            if logo is not None and self.kb.logo_index is not None:
                evidence.logo_found = True
                match = self.kb.logo_index.best(self.logos.embed(logo))
                if match is not None:
                    evidence.logo_brand_id, evidence.logo_similarity = match
        text_brand = evidence.summary.get("brand")
        if has_brand(text_brand):
            evidence.text_brand = text_brand.strip()
            evidence.text_brand_ids = sorted(self.kb.brands_for_alias(evidence.text_brand))
        return evidence

    def _identified(self, evidence: KPDEvidence, threshold: float) -> list[str]:
        """Return Algorithm 1 lines 5-8: the logo brand if it clears the threshold, otherwise the alias matches."""
        similarity = evidence.logo_similarity
        if evidence.logo_brand_id is not None and similarity is not None and similarity >= threshold:
            return [evidence.logo_brand_id]
        return list(evidence.text_brand_ids)

    def decide(self, evidence: KPDEvidence, page: Page, setting: dict[str, Any]) -> Decision:
        threshold = float(setting["logo_threshold"])
        raw: dict[str, Any] = {
            "domain": evidence.domain,
            "summary_response": evidence.summary_text,
            "model": evidence.model,
            "is_crp": evidence.is_crp,
            "logo_found": evidence.logo_found,
            "logo_brand": self.kb[evidence.logo_brand_id].name if evidence.logo_brand_id else None,
            "logo_similarity": evidence.logo_similarity,
            "text_brand": evidence.text_brand,
            "text_brands_matched": [self.kb[b].name for b in evidence.text_brand_ids],
        }
        if evidence.error is not None:
            return Decision(None, None, "page failed", raw, evidence.error)
        identified = self._identified(evidence, threshold)
        names = [self.kb[b].name for b in identified]
        # The brands are recorded for pages that stop at the CRP gate so the binding control can use them; the verdict below ignores them.
        raw["brands_ignoring_crp_gate"] = names
        if not evidence.is_crp:
            raw["brands"] = []
            return Decision(BENIGN, None, "The page was classified as not requesting credentials.", raw)
        raw["brands"] = names
        if not identified:
            return Decision(BENIGN, None, "No known brand was identified from the logo or the text.", raw)
        for brand_id in identified:
            if evidence.domain in self.kb[brand_id].domains:
                name = self.kb[brand_id].name
                return Decision(BENIGN, name, f"The page's domain is a known domain of {name}.", raw)
        return Decision(PHISH, names[0], f"The page presents {names[0]} but its domain is not a known domain of that brand.", raw)
