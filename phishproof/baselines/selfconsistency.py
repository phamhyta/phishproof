"""B3 — self-consistency (wang2023selfconsistency): one model, k samples, agreement = score.

The k samples require model calls, so the runner (Phase 5) draws them; this module just
scores a set of sampled labels. Score = fraction agreeing with the modal label.
"""

from __future__ import annotations

from collections import Counter

from ..schema import Label


def self_consistency_score(sampled_labels: list[Label]) -> float:
    if not sampled_labels:
        return 0.0
    counts = Counter(sampled_labels)
    return counts.most_common(1)[0][1] / len(sampled_labels)


def self_consistency_label(sampled_labels: list[Label]) -> Label:
    return Counter(sampled_labels).most_common(1)[0][0]
