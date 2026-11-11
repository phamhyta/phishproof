"""Group near-duplicate captures, the unit over which the policy counts its errors.

Two pages join a group if their HTML or screenshot bytes are identical, or if the 64-bit
perceptual hashes of their screenshots are within a small Hamming distance.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..schema import PageRecord

DEFAULT_HAMMING_RADIUS = 3


@dataclass(frozen=True)
class Fingerprint:
    page_id: str
    html_sha: str | None
    screenshot_sha: str | None
    perceptual_hash: int | None


def _sha256(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference_hash(path: str | None, size: int = 8) -> int | None:
    if not path or not Path(path).exists():
        return None
    try:
        image = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except (OSError, UnidentifiedImageError):
        return None
    pixels = np.asarray(image, dtype=np.int16)
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).flatten():
        value = (value << 1) | int(bit)
    return value


def fingerprint(page: PageRecord) -> Fingerprint:
    return Fingerprint(
        page_id=page.page_id,
        html_sha=_sha256(page.dom_html_path),
        screenshot_sha=_sha256(page.screenshot_path),
        perceptual_hash=difference_hash(page.screenshot_path),
    )


class _DisjointSet:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: str, second: str) -> None:
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            # The smaller id stays the root, so group names do not depend on input order.
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def _popcount(values: np.ndarray) -> np.ndarray:
    values = values - ((values >> np.uint64(1)) & np.uint64(0x5555555555555555))
    values = (values & np.uint64(0x3333333333333333)) + (
        (values >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    values = (values + (values >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((values * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def group_fingerprints(
    prints: Sequence[Fingerprint], hamming_radius: int = DEFAULT_HAMMING_RADIUS
) -> dict[str, str]:
    """Map each page id to its group id, the smallest page id in the group."""
    groups = _DisjointSet(item.page_id for item in prints)
    first_seen: dict[tuple[str, str], str] = {}
    for item in prints:
        for field in ("html_sha", "screenshot_sha"):
            value = getattr(item, field)
            if value:
                groups.union(item.page_id, first_seen.setdefault((field, value), item.page_id))

    hashed = [item for item in prints if item.perceptual_hash is not None]
    hashes = np.array([item.perceptual_hash for item in hashed], dtype=np.uint64)
    for index in range(len(hashed) - 1):
        distances = _popcount(hashes[index + 1 :] ^ hashes[index])
        for offset in np.flatnonzero(distances <= hamming_radius):
            groups.union(hashed[index].page_id, hashed[index + 1 + int(offset)].page_id)
    return {item.page_id: groups.find(item.page_id) for item in prints}


def duplicate_groups(
    pages: Sequence[PageRecord], hamming_radius: int = DEFAULT_HAMMING_RADIUS
) -> dict[str, str]:
    return group_fingerprints([fingerprint(page) for page in pages], hamming_radius)


def without_test_duplicates(
    calibration: Sequence[PageRecord],
    test_page_ids: Iterable[str],
    groups: Mapping[str, str],
) -> list[PageRecord]:
    """Remove every calibration page whose duplicate group also occurs in a test split."""
    test_groups = {groups[page_id] for page_id in test_page_ids}
    return [page for page in calibration if groups[page.page_id] not in test_groups]
