"""Pre-flight bug audit: run the whole non-model pipeline over every benchmark page.

The expensive run calls models; a crash midway wastes ~1.5h. This script exercises
everything that does NOT need a model — render_context, the DOM/cert/redirect tools, the
logo crop, the consistency check, and grounding of typed cues — over EVERY page, and
reports any exception with the offending page. Run it (and fix what it finds) before Pass 1.

Usage:  .venv/bin/python scripts/preflight.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.agents.page_context import render_context
from phishproof.data_io import read_manifest
from phishproof.schema import Cue, CueType
from phishproof.tools.consistency import consistency_for_brand
from phishproof.tools.dom import form_action_domains, has_credential_intent
from phishproof.tools.logo_brand import crop_logo, parse_logo_boxes
from phishproof.tools.registry import GroundingContext, ground_cue


def check_page(page) -> list[str]:
    errs = []
    brand = page.brand or "example"
    checks = [
        ("render_context", lambda: render_context(page)),
        ("form_action_domains", lambda: form_action_domains(page)),
        ("has_credential_intent", lambda: has_credential_intent(page)),
        ("parse_logo_boxes", lambda: parse_logo_boxes(page)),
        ("crop_logo", lambda: crop_logo(page)),
        ("consistency", lambda: consistency_for_brand(brand, page)),
        # ground every non-perceptual cue type (logo CLIP needs a model -> skipped here)
        ("ground.brand", lambda: ground_cue(Cue(type=CueType.BRAND_CLAIM, value=brand), page)),
        ("ground.form", lambda: ground_cue(
            Cue(type=CueType.FORM_ACTION_DOMAIN, value="example.com"), page)),
        ("ground.cred", lambda: ground_cue(
            Cue(type=CueType.CREDENTIAL_INTENT, value="yes"), page)),
        ("ground.consistency", lambda: ground_cue(
            Cue(type=CueType.BRAND_DOMAIN_CONSISTENCY, value="inconsistent", raw_value=brand), page)),
    ]
    for name, fn in checks:
        try:
            fn()
        except Exception:  # noqa: BLE001
            errs.append(f"{name}: {traceback.format_exc().splitlines()[-1]}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    args = ap.parse_args()

    pages = (read_manifest(args.data / "calibration.jsonl")
             + read_manifest(args.data / "test.jsonl"))
    print(f"preflight over {len(pages)} pages...")
    n_bad = 0
    for i, page in enumerate(pages, 1):
        errs = check_page(page)
        if errs:
            n_bad += 1
            print(f"[CRASH] {page.page_id} ({page.url[:50]})")
            for e in errs:
                print(f"        {e}")
        if i % 200 == 0:
            print(f"  ...{i}/{len(pages)} ({n_bad} bad)")
    print(f"\n{'[ok] zero crashes' if n_bad == 0 else f'[FAIL] {n_bad} pages crashed'}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
