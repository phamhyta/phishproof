"""D1 — Phishpedia (lin2021phishpedia): reference-based visual brand matcher.

Standalone detection and brand grounding both read from a precomputed JSONL cache
produced by the containerized detectron2 run (see scripts/import_phishpedia_d1.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from ...schema import Label, PageRecord
from ..brands import canonical_brand


def load_phishpedia_cache(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["page_id"]] = row
    return rows


class PhishpediaDetector:
    name = "D1-Phishpedia"

    def __init__(self, cache_path: Path | str = Path("results/phishpedia_d1.jsonl")) -> None:
        self._cache = load_phishpedia_cache(Path(cache_path))

    def derive_brand(self, page: PageRecord) -> str | None:
        row = self._cache.get(page.page_id)
        if not row:
            return None
        brand = row.get("brand")
        return canonical_brand(brand) if brand else None

    def predict(self, page: PageRecord) -> Label:
        row = self._cache.get(page.page_id)
        if not row:
            return Label.BENIGN
        return Label.PHISH if row.get("verdict") == "phish" else Label.BENIGN
