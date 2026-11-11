"""The conf+GEA combiner: logistic regression of correctness on vision confidence and GEA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


class ConfidenceAgreementCombiner:
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1.0)

    def fit(
        self, vision: Sequence[float], gea: Sequence[float], correct: Sequence[bool]
    ) -> ConfidenceAgreementCombiner:
        self.model.fit(np.column_stack([vision, gea]), np.asarray(correct, dtype=int))
        return self

    def score(self, vision: Sequence[float], gea: Sequence[float]) -> np.ndarray:
        return self.model.predict_proba(np.column_stack([vision, gea]))[:, 1]
