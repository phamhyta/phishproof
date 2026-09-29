"""Per-class isotonic calibrator: separate kappa for predicted-phish vs predicted-benign.

The single global isotonic calibrator (isotonic.py) maps GEA -> P(correct) with ONE
monotone curve for every page. Diagnostic on the test bundle showed this is the root of
the Cov99 loss to B6: phish-correct pages are systematically under-confident (only ~12.5%
reach trust>0.8) while benign-correct pages sit higher, so a single curve cannot push the
confident phish cluster up without also inflating weak benign pages.

PerClassCalibrator fits two isotonic curves keyed on the panel's VERDICT (not the gold
label, which is unavailable at inference) -- one for pages the panel calls phish and one
for pages it calls benign -- so each cluster gets its own GEA->P(correct) mapping. This is
still calibration-scale fitting on the held-out split, order-preserving WITHIN each class.
"""

from __future__ import annotations

from .isotonic import IsotonicCalibrator


class PerClassCalibrator:
    def __init__(self) -> None:
        self._by_verdict: dict[str, IsotonicCalibrator] = {}
        self._fallback = IsotonicCalibrator()

    def fit(
        self,
        scores: list[float],
        correct: list[bool],
        verdicts: list[str],
    ) -> "PerClassCalibrator":
        self._fallback.fit(scores, correct)
        for v in set(verdicts):
            xs = [s for s, vv in zip(scores, verdicts) if vv == v]
            ys = [c for c, vv in zip(correct, verdicts) if vv == v]
            self._by_verdict[v] = IsotonicCalibrator().fit(xs, ys)
        return self

    def predict(self, scores: list[float], verdicts: list[str]) -> list[float]:
        out = []
        for s, v in zip(scores, verdicts):
            cal = self._by_verdict.get(v, self._fallback)
            out.append(cal(s))
        return out

    def to_dict(self) -> dict:
        return {
            "kind": "per_class",
            "by_verdict": {v: c.to_dict() for v, c in self._by_verdict.items()},
            "fallback": self._fallback.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PerClassCalibrator":
        c = cls()
        c._by_verdict = {
            v: IsotonicCalibrator.from_dict(cc) for v, cc in d["by_verdict"].items()
        }
        c._fallback = IsotonicCalibrator.from_dict(d["fallback"])
        return c
