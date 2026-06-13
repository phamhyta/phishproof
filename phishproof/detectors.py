from __future__ import annotations

from typing import Protocol
from urllib.parse import urlparse

from .types import PageRecord, Prediction


class Detector(Protocol):
    def predict(self, page: PageRecord) -> Prediction: ...


class UrlSignalDetector:
    """Small baseline used to keep the public scaffold runnable."""

    def __init__(self, threshold: float = 0.55) -> None:
        self.threshold = threshold

    def predict(self, page: PageRecord) -> Prediction:
        score, reasons = self._score(page)
        label = "phishing" if score >= self.threshold else "benign"
        return Prediction(page.id, label, score, tuple(reasons))

    def _score(self, page: PageRecord) -> tuple[float, list[str]]:
        parsed = urlparse(page.url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        text = f"{host} {path} {page.title.lower()}"

        reasons: list[str] = []
        score = 0.12

        if any(term in text for term in ("login", "verify", "wallet", "password", "account")):
            score += 0.24
            reasons.append("credential-oriented wording")

        if parsed.scheme != "https":
            score += 0.18
            reasons.append("non-https URL")

        if host.count(".") >= 3:
            score += 0.15
            reasons.append("deeply nested host")

        if any(ch.isdigit() for ch in host.split(".")[0]):
            score += 0.09
            reasons.append("digits in host label")

        if "-" in host:
            score += 0.08
            reasons.append("hyphenated host")

        return min(score, 0.99), reasons or ["no strong URL signal"]
