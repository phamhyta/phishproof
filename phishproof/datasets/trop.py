"""Loader for KnowPhish TR-OP (``openphish_5000``), whose folders are ``<brand>+<host>+<date>+<n>``."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..schema import Label, PageRecord
from .phishpedia import load_folder

_NOT_A_BRAND = {"", "?", "-", "_", "unknown"}


def brand_from_folder(name: str) -> str | None:
    head = name.partition("+")[0].strip()
    return None if head in _NOT_A_BRAND else head


def iter_trop(root: Path) -> Iterator[PageRecord]:
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        record = load_folder(folder, Label.PHISH, source="trop")
        if record is not None:
            yield record.model_copy(update={"brand": brand_from_folder(folder.name)})
