"""The fixed prompt given to every panel member and the response format it must follow."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..schema import ACTIVE_CUE_TYPES, CueType, Label

_CUE_DOC = """\
- brand_claim: the brand this page presents itself as (its logo/title/branding), whether \
legitimately or by impersonation, e.g. "PayPal". Omit if the page shows no clear brand.
- form_action_domain: the registrable domain (eTLD+1) a login/credential form posts to.
- credential_intent: whether the page asks for credentials (login/password/payment). "yes"/"no". \
This alone is NEUTRAL — legitimate sites also have login forms.
- logo_brand: the brand of the most prominent rendered logo. Omit if no clear logo."""

SYSTEM_PROMPT = f"""You are a security analyst deciding whether a web page is PHISHING or \
LEGITIMATE (benign). Decide by comparing the brand the page presents against the domain it \
runs on — apply this rule directly:

1. Identify the brand the page presents (from its logo, title, and text). It may have none.
2. Find the page's own domain ("URL_host_etld1") and the domains its forms post to.
3. DECISION RULE:
   - PHISHING: the page presents a recognizable brand (e.g. PayPal, Microsoft, a bank) but \
its domain / form-action domain is NOT that brand's official domain — i.e. brand and domain \
are INCONSISTENT. A login/credential form on such a mismatched domain is a strong phishing \
signal.
   - BENIGN: the page's branding matches its domain (a brand on its own official domain), OR \
the page presents no impersonated brand at all (an ordinary site on its own domain). A login \
form is fine when the domain matches the brand.

Be balanced: do not call a page phishing just because it has a login form, and do not call it \
benign just because it looks polished — a convincing brand replica on an unrelated domain is \
exactly what phishing is. Weigh the brand-vs-domain match, then decide.

Justify the verdict with EVIDENCE CUES drawn ONLY from this fixed schema:

{_CUE_DOC}

Rules:
- Emit a cue only when you actually observe it on this page; do not invent cues to fill the list.
- Values must be exact (a real domain, a real brand name).
- Output STRICT JSON only, matching the schema. No prose outside the JSON."""

USER_TEMPLATE = """Decide if this page is phishing or legitimate, weighing whether its brand \
is consistent with its domain. Return the JSON only.

{context}

Return JSON: {{"verdict": "phish"|"benign", "confidence": 0..1, "cues": [{{"type": <cue type>, "value": <string>}}]}}"""

class RawCue(BaseModel):
    type: CueType
    value: str


class RawAgentResponse(BaseModel):
    verdict: Label
    cues: list[RawCue] = Field(default_factory=list)
    confidence: float | None = None


# Passed as ``response_format`` to providers that enforce a JSON schema.
STRICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "panel_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["verdict", "confidence", "cues"],
            "properties": {
                "verdict": {"type": "string", "enum": [label.value for label in Label]},
                "confidence": {"type": "number"},
                "cues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "value"],
                        "properties": {
                            "type": {"type": "string", "enum": [t.value for t in ACTIVE_CUE_TYPES]},
                            "value": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}


def build_user_prompt(context: str) -> str:
    return USER_TEMPLATE.format(context=context)
