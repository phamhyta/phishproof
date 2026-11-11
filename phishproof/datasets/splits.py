"""Sample a benchmark from ingested pages and split it into calibration and test sets."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from ..schema import Label, PageRecord
from ..verification.binding import squeeze
from ..verification.brands import canonical_brand
from ..verification.dom import has_credential_intent
from ..verification.urls import hostname


@dataclass(frozen=True)
class BenchmarkSpec:
    n_ordinary: int
    n_lookalike: int
    n_benign: int
    calibration_share: float = 0.15
    max_per_brand: int = 30
    seed: int = 21


def is_lookalike(page: PageRecord) -> bool:
    """True if the page's host contains its brand."""
    brand = squeeze(canonical_brand(page.brand) or "")
    host = squeeze(hostname(page.host_url) or "")
    return bool(brand) and brand in host


def credential_login_pages(pages: list[PageRecord]) -> list[PageRecord]:
    """Pages whose DOM asks for a credential, the benign side of a benchmark."""
    return [page for page in pages if has_credential_intent(page)]


def capped_sample(
    pages: list[PageRecord], count: int, max_per_brand: int, rng: random.Random
) -> list[PageRecord]:
    """Sample with a per-brand limit, and backfill from the remainder if too few pages are left."""
    by_brand: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        by_brand[canonical_brand(page.brand) or "_"].append(page)
    capped: list[PageRecord] = []
    for brand_pages in by_brand.values():
        rng.shuffle(brand_pages)
        capped.extend(brand_pages[:max_per_brand])
    rng.shuffle(capped)
    if len(capped) < count:
        chosen = {page.page_id for page in capped}
        rest = [page for page in pages if page.page_id not in chosen]
        rng.shuffle(rest)
        capped.extend(rest[: count - len(capped)])
    return capped[:count]


def stratified_split(
    pages: list[PageRecord], calibration_share: float, rng: random.Random
) -> tuple[list[PageRecord], list[PageRecord]]:
    calibration: list[PageRecord] = []
    test: list[PageRecord] = []
    for label in Label:
        members = [page for page in pages if page.label is label]
        rng.shuffle(members)
        cut = round(len(members) * calibration_share)
        calibration.extend(page.model_copy(update={"split": "calibration"}) for page in members[:cut])
        test.extend(page.model_copy(update={"split": "test"}) for page in members[cut:])
    rng.shuffle(calibration)
    rng.shuffle(test)
    return calibration, test


def build_benchmark(
    phish: list[PageRecord], benign: list[PageRecord], spec: BenchmarkSpec
) -> tuple[list[PageRecord], list[PageRecord]]:
    """Mix ordinary and lookalike phishing pages with benign pages, then split the result."""
    rng = random.Random(spec.seed)
    ordinary = [page for page in phish if not is_lookalike(page)]
    lookalike = [
        page.model_copy(update={"source": "lookalike"}) for page in phish if is_lookalike(page)
    ]
    pool = (
        capped_sample(ordinary, spec.n_ordinary, spec.max_per_brand, rng)
        + capped_sample(lookalike, spec.n_lookalike, spec.max_per_brand, rng)
        + rng.sample(benign, spec.n_benign)
    )
    return stratified_split(pool, spec.calibration_share, rng)
