"""Loader for Phishpedia-style capture folders: ``<root>/<page>/{html.txt, shot.png, info.txt}``."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from ..schema import Label, PageRecord
from .manifest import slug


def parse_info(path: Path) -> dict:
    """``info.txt`` holds a dict literal for phishing pages and a bare URL for benign ones."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return {"url": text.splitlines()[0].strip()}
    return value if isinstance(value, dict) else {"url": str(value)}


def load_folder(folder: Path, label: Label, source: str = "phishpedia") -> PageRecord | None:
    html, shot, info = folder / "html.txt", folder / "shot.png", folder / "info.txt"
    if not (html.exists() and shot.exists()):
        return None
    meta = parse_info(info) if info.exists() else {}
    return PageRecord(
        page_id=slug(folder.name),
        url=meta.get("url") or f"http://{folder.name}",
        label=label,
        dom_html_path=str(html),
        screenshot_path=str(shot),
        raw_dir=str(folder),
        brand=(meta.get("brand") or None) if label is Label.PHISH else None,
        source=source,
    )


def iter_phishpedia(root: Path, label: Label) -> Iterator[PageRecord]:
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        record = load_folder(folder, label)
        if record is not None:
            yield record
