"""Per-action threshold selection by a fixed-sequence exact binomial scan over duplicate groups.

Levels are scanned from high to low. A level is kept while the one-sided Clopper-Pearson upper
limit on the action's group error rate stays within its budget, and the scan stops at the first
level that fails. The threshold is the last level kept. If the first level already fails, the
action is disabled and its pages abstain.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from scipy.stats import beta

from ..schema import Label, PageEvidence
from .ranking import reaches


@dataclass(frozen=True)
class ScanResult:
    level: float | None
    n_groups: int
    failures: int
    upper_limit: float


def clopper_pearson_upper(failures: int, n: int, delta: float) -> float:
    if n == 0 or failures >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, failures + 1, n - failures))


def score_lattice(top: float = 1.0, bottom: float = 0.5, step: float = 0.01) -> list[float]:
    count = round((top - bottom) / step)
    return [round(top - index * step, 10) for index in range(count + 1)]


def scan(
    wrong_pages: Iterable[tuple[str, float]],
    n_groups: int,
    alpha: float,
    delta: float,
    levels: Sequence[float] | None = None,
) -> ScanResult:
    """Select a level from the (group, score) pairs of pages on which the action would be wrong."""
    wrong_pages = list(wrong_pages)
    kept = ScanResult(None, n_groups, 0, 1.0)
    for level in levels if levels is not None else score_lattice():
        failing_groups = {group for group, score in wrong_pages if reaches(score, level)}
        limit = clopper_pearson_upper(len(failing_groups), n_groups, delta)
        if limit > alpha:
            break
        kept = ScanResult(level, n_groups, len(failing_groups), limit)
    return kept


def _groups_of_class(
    evidence: Sequence[PageEvidence], groups: Mapping[str, str], label: Label
) -> int:
    return len({groups[page.page_id] for page in evidence if page.label is label})


def scan_block(
    evidence: Sequence[PageEvidence], groups: Mapping[str, str], alpha: float, delta: float
) -> ScanResult:
    """A block is wrong on a benign page. Failures are counted over benign groups."""
    wrong = [
        (groups[page.page_id], page.rank)
        for page in evidence
        if page.label is Label.BENIGN
        and page.verdict is Label.PHISH
        and page.complete
        and page.gate_passed
    ]
    return scan(wrong, _groups_of_class(evidence, groups, Label.BENIGN), alpha, delta)


def scan_clear(
    evidence: Sequence[PageEvidence], groups: Mapping[str, str], alpha: float, delta: float
) -> ScanResult:
    """A clear is wrong on a phishing page. Failures are counted over phishing groups."""
    wrong = [
        (groups[page.page_id], page.rank)
        for page in evidence
        if page.label is Label.PHISH
        and page.verdict is Label.BENIGN
        and page.complete
        and page.gate_passed
        and page.bound
    ]
    return scan(wrong, _groups_of_class(evidence, groups, Label.PHISH), alpha, delta)
