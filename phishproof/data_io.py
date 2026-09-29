"""Read/write PhishSel manifests (JSONL of PageRecord) and parse Phishpedia folders."""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from .schema import Label, PageRecord


def slug(name: str) -> str:
    """Stable, filesystem-safe page id from a folder name."""
    base = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:60]
    h = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{base}-{h}"


def parse_phishpedia_info(info_path: Path) -> dict:
    """Parse info.txt. Phish set: a Python dict literal (url, brand, ...). Benign set:
    a bare URL string. Returns a dict either way ({} if unparseable/empty)."""
    text = info_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {}
    try:
        val = ast.literal_eval(text)
        return val if isinstance(val, dict) else {"url": str(val)}
    except (ValueError, SyntaxError):
        return {"url": text.splitlines()[0].strip()}  # plain URL (benign)


def phishpedia_folder_to_record(folder: Path, label: Label) -> PageRecord | None:
    """One Phishpedia sample folder -> PageRecord, or None if DOM/screenshot missing.

    Requires html.txt + shot.png (PhishProof needs the DOM and the screenshot). The URL
    comes from info.txt when present, else from the folder name (benign folders are named
    by domain). brand is taken from info.txt only for the phish set.
    """
    html = folder / "html.txt"
    shot = folder / "shot.png"
    if not (html.exists() and shot.exists()):
        return None
    info = folder / "info.txt"
    meta = parse_phishpedia_info(info) if info.exists() else {}
    url = meta.get("url") or f"http://{folder.name}"
    return PageRecord(
        page_id=slug(folder.name),
        url=url,
        label=label,
        dom_html_path=str(html),
        screenshot_path=str(shot),
        raw_dir=str(folder),
        brand=(meta.get("brand") or None) if label is Label.PHISH else None,
        source="phishpedia",
    )


def iter_phishpedia(raw_dir: Path, label: Label) -> Iterator[PageRecord]:
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        rec = phishpedia_folder_to_record(folder, label)
        if rec is not None:
            yield rec


def write_manifest(records: list[PageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def read_manifest(path: Path) -> list[PageRecord]:
    with Path(path).open(encoding="utf-8") as f:
        return [PageRecord.model_validate_json(line) for line in f if line.strip()]
