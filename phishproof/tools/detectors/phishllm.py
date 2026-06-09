"""D3 — PhishLLM (tan2024phishllm): an LLM brand-and-domain checker, used as a standalone
full-coverage detector for tab_detect.

Faithful to PhishLLM's core logic: (1) recognise the brand the page presents, (2) decide
whether the page solicits credentials, (3) check whether the page's domain is the brand's
legitimate domain — phishing iff a credential page impersonates a brand on a domain that is
not the brand's own. One GPT-4o call per page over the page context (+ screenshot).
"""

from __future__ import annotations

import json

from ...config import AgentConfig
from ...schema import Label, PageRecord
from ...agents.client import ChatClient
from ...agents.page_context import render_context

SYSTEM = """You are PhishLLM, a brand-and-domain phishing checker. For the web page:
1. Identify the brand it presents (logo/title/text), or 'none'.
2. Decide if it solicits credentials/payment (login, password, card).
3. Determine the page's own registrable domain and whether that domain is the brand's
   OFFICIAL domain.
A page is PHISHING iff it impersonates a brand and asks for credentials while its domain is
NOT that brand's official domain. A brand on its own domain, or a page with no brand
impersonation, is BENIGN. Output STRICT JSON only:
{"brand": <str>, "credential": <bool>, "domain_is_official": <bool>, "verdict": "phish"|"benign"}."""


class PhishLLMDetector:
    name = "D3-PhishLLM"

    def __init__(self, client: ChatClient | None = None, model: str = "gpt-4o",
                 use_vision: bool = True) -> None:
        self.client = client or ChatClient()
        self.cfg = AgentConfig(id="phishllm", provider="openai", model=model,
                               modality="vision" if use_vision else "text", detail="low")
        self.use_vision = use_vision

    def predict(self, page: PageRecord) -> Label:
        user = ("Check this page.\n\n" + render_context(page)
                + '\n\nReturn JSON {"brand","credential","domain_is_official","verdict"}.')
        image = page.screenshot_path if self.use_vision else None
        raw = self.client.complete_json(self.cfg, SYSTEM, user, image)
        try:
            d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            return Label(d.get("verdict", "benign"))
        except Exception:  # noqa: BLE001
            return Label.BENIGN
