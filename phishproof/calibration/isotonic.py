"""Isotonic calibrator kappa: GEA score -> P(verdict correct) (C4, method.tex §sec:select).

Monotonic (order-preserving), so it leaves the score's ranking — hence the
risk--coverage curve — intact while turning it into a probability of correctness.
Fit on the labeled calibration split only.

Implementation note: we fit with sklearn's IsotonicRegression, then keep only its
(x, y) step points and evaluate with np.interp. That makes the calibrator trivially
serializable (to_dict / from_dict) and independent of sklearn internals at predict time.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._kind: str | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._const: float = 0.5

    def fit(self, scores: list[float], correct: list[bool]) -> "IsotonicCalibrator":
        x = np.asarray(scores, dtype=float)
        y = np.asarray(correct, dtype=float)
        if len(x) < 2 or len(np.unique(x)) < 2:
            self._kind = "const"
            self._const = float(y.mean()) if len(y) else 0.5
            return self
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
        self._x = np.asarray(ir.X_thresholds_, dtype=float)
        self._y = np.asarray(ir.y_thresholds_, dtype=float)
        self._kind = "isotonic"
        return self

    def predict(self, scores: list[float]) -> list[float]:
        if self._kind is None:
            raise RuntimeError("calibrator not fitted")
        s = np.asarray(scores, dtype=float)
        if self._kind == "const":
            return [self._const] * len(s)
        # np.interp clamps to the endpoint y-values outside [x[0], x[-1]] (== "clip").
        return [float(v) for v in np.interp(s, self._x, self._y)]

    def __call__(self, score: float) -> float:
        return self.predict([score])[0]

    def to_dict(self) -> dict:
        if self._kind == "const":
            return {"kind": "const", "value": self._const}
        return {"kind": "isotonic", "x": self._x.tolist(), "y": self._y.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        c = cls()
        if d["kind"] == "const":
            c._kind, c._const = "const", float(d["value"])
        else:
            c._kind = "isotonic"
            c._x = np.asarray(d["x"], dtype=float)
            c._y = np.asarray(d["y"], dtype=float)
        return c
