"""Loader for APWG ``phishing4190`` (PhishingEval, Zenodo 14668190).

The corpus lists its captures in ``phishing4190_2.csv`` and keeps no original URL, so each page
gets a URL built from its recorded domain.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ..schema import Label, PageRecord
from .manifest import slug


def iter_apwg(root: Path) -> Iterator[PageRecord]:
    with (root / "phishing4190_2.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            screenshot_name = (row.get("scr_path") or "").strip()
            html_name = (row.get("html_path") or "").strip()
            if not (screenshot_name and html_name):
                continue
            screenshot, html = (root / screenshot_name).resolve(), (root / html_name).resolve()
            if not (screenshot.exists() and html.exists()):
                continue
            page_id = slug(Path(html_name).stem)
            domain = (row.get("domain") or "").strip()
            yield PageRecord(
                page_id=page_id,
                url=f"http://{domain}/" if domain else f"http://apwg/{page_id}",
                label=Label.PHISH,
                dom_html_path=str(html),
                screenshot_path=str(screenshot),
                raw_dir=str(screenshot.parent),
                brand=(row.get("brand") or "").strip() or None,
                source="apwg",
            )
