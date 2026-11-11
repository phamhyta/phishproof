"""Multimodal retriever logic that does not need a GPU.

Implements the data preparation of PhishAgent App. A.3 (training-page selection, the grounding step, positive pairs with
one pair per logo variant, two rounds of negative sampling, the 9:1 split), the "also known as" brand text of Sec. 4.3,
and top-k retrieval from the dot-product scores of Sec. 4.3 Inference. The encoders are in clip_retriever.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .base import Page
from .kb import Brand, KnowledgeBase


@dataclass(frozen=True)
class Hit:
    brand_id: str
    score: float


@dataclass(frozen=True)
class BrandEntry:
    """One logo variant of a brand, treated as a separate brand as Sec. 4.3 says; logo is None for a logo-less brand."""

    brand_id: str
    text: str
    logo: str | None


@dataclass(frozen=True)
class Pair:
    page_id: str
    group_id: str
    positive: int
    negatives: tuple[int, ...]


def brand_text(brand: Brand) -> str:
    """Return the Sec. 4.3 string: {name}, also known as {aka1}, ..., {akaA}."""
    others = [alias for alias in brand.aliases if alias != brand.name]
    return f"{brand.name}, also known as {', '.join(others)}" if others else brand.name


def brand_entries(kb: KnowledgeBase) -> list[BrandEntry]:
    entries: list[BrandEntry] = []
    for brand in kb.brands.values():
        text = brand_text(brand)
        if brand.logos:
            entries.extend(BrandEntry(brand.brand_id, text, logo) for logo in brand.logos)
        else:
            entries.append(BrandEntry(brand.brand_id, text, None))
    return entries


def ranked_brands(scores: np.ndarray, entry_brand_ids: Sequence[str], limit: int) -> list[Hit]:
    """Return up to limit distinct brands by score, keeping each brand's best logo variant."""
    # logo variants are collapsed so that top-k holds k distinct brands, because the paper retrieves "the top k brands" yet treats variants as separate brands.
    hits: list[Hit] = []
    seen: set[str] = set()
    for index in np.argsort(-scores, kind="stable"):
        brand_id = entry_brand_ids[int(index)]
        if brand_id in seen:
            continue
        seen.add(brand_id)
        hits.append(Hit(brand_id, float(scores[int(index)])))
        if len(hits) == limit:
            break
    return hits


def select_training_pages(pool: Iterable[Page], excluded: Iterable[Page], n_pages: int, seed: int) -> list[Page]:
    """Sample phishing pages outside every excluded page id and duplicate group (App. A.3)."""
    excluded = list(excluded)
    excluded_ids = {page.page_id for page in excluded}
    excluded_groups = {page.group_id for page in excluded}
    candidates = sorted(
        (
            page
            for page in pool
            if page.label == 1 and page.page_id not in excluded_ids and page.group_id not in excluded_groups
        ),
        key=lambda page: page.page_id,
    )
    if len(candidates) < n_pages:
        raise ValueError(f"only {len(candidates)} eligible phishing pages, {n_pages} requested")
    return random.Random(seed).sample(candidates, n_pages)


def ground_pages(pages: Iterable[Page], labels: dict[str, str], kb: KnowledgeBase) -> list[tuple[Page, str]]:
    """Keep pages whose target-brand label matches exactly one brand in the knowledge base (App. A.3)."""
    # the paper's unnamed "string match algorithm" is the alias match of KnowledgeBase.brands_for_alias.
    grounded: list[tuple[Page, str]] = []
    for page in pages:
        matches = kb.brands_for_alias(labels.get(page.page_id, ""))
        if len(matches) == 1:
            grounded.append((page, next(iter(matches))))
    return grounded


def build_pairs(
    grounded: Sequence[tuple[Page, str]],
    entries: Sequence[BrandEntry],
    n_negatives: int,
    rounds: int,
    seed: int,
) -> list[Pair]:
    """Create one positive per logo variant and sample negatives once per round (App. A.3)."""
    rng = random.Random(seed)
    by_brand: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        by_brand[entry.brand_id].append(index)
    pairs: list[Pair] = []
    for page, brand_id in grounded:
        pool_size = len(entries) - len(by_brand[brand_id])
        if pool_size < n_negatives:
            raise ValueError(f"knowledge base has only {pool_size} negatives for brand {brand_id}")
        for positive in by_brand[brand_id]:
            for _ in range(rounds):
                negatives: set[int] = set()
                while len(negatives) < n_negatives:
                    candidate = rng.randrange(len(entries))
                    if entries[candidate].brand_id != brand_id:
                        negatives.add(candidate)
                pairs.append(Pair(page.page_id, page.group_id, positive, tuple(sorted(negatives))))
    return pairs


def split_pairs(pairs: Sequence[Pair], val_fraction: float, seed: int) -> tuple[list[Pair], list[Pair]]:
    """Split pairs 9:1 by duplicate group."""
    # the 9:1 split is by duplicate group so that near-duplicate pages are not in both parts, because the paper does not say how the pairs are divided.
    groups = sorted({pair.group_id for pair in pairs})
    random.Random(seed).shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    val_groups = set(groups[:n_val])
    train = [pair for pair in pairs if pair.group_id not in val_groups]
    val = [pair for pair in pairs if pair.group_id in val_groups]
    return train, val
