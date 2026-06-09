"""Phase 2 smoke test: run grounding tools on real Phishpedia pages.

Exercises DOM (form-action, credential-intent), brand-claim (gold stand-in), cert/redirect
(N/A), and logo cropping on the first few records of manifest_phish.jsonl.

Run:  .venv/bin/python scripts/check_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import read_manifest
from phishproof.schema import Cue, CueType
from phishproof.tools import dom, logo_brand
from phishproof.tools.brands import canonical_brand
from phishproof.tools.registry import GroundingContext, ground_cue


class FakeEmbedder:
    """Stand-in for CLIP so the logo path runs without torch installed."""

    def similarity(self, crop, brand: str) -> float:
        # crude proxy: bigger crop -> assume a clearer logo; just exercises the path
        return round(min(1.0, (crop.size[0] * crop.size[1]) / 50000.0), 3)


def main() -> int:
    manifest = Path("data/phishsel/manifest_phish.jsonl")
    if not manifest.exists():
        print(f"[FAIL] {manifest} not found — run ingest_phishpedia.py first")
        return 1

    records = read_manifest(manifest)[:5]
    ctx = GroundingContext(logo_embedder=FakeEmbedder())
    print(f"Testing tools on {len(records)} real phishing pages\n")

    for r in records:
        print(f"--- {r.page_id}")
        print(f"    url={r.url}")
        print(f"    brand(gold)={r.brand!r} -> canonical={canonical_brand(r.brand)!r}")

        fads = dom.form_action_domains(r)
        cred = dom.has_credential_intent(r)
        print(f"    form_action_domains={sorted(fads)}")
        print(f"    credential_intent={cred}")

        crop = logo_brand.crop_logo(r)
        print(f"    logo_crop={'%dx%d' % crop.size if crop else None}")

        # Ground a few cues, including correct and deliberately wrong claims.
        cues = [
            Cue(type=CueType.BRAND_CLAIM, value=r.brand or "unknown"),
            Cue(type=CueType.BRAND_CLAIM, value="DefinitelyWrongBrand"),
            Cue(type=CueType.CREDENTIAL_INTENT, value="yes"),
            Cue(type=CueType.CERTIFICATE_ORG, value="PayPal"),  # N/A on Phishpedia
        ]
        if fads:
            cues.append(Cue(type=CueType.FORM_ACTION_DOMAIN, value=sorted(fads)[0]))
            cues.append(Cue(type=CueType.FORM_ACTION_DOMAIN, value="evil-not-here.com"))
        if crop:
            cues.append(Cue(type=CueType.LOGO_BRAND, value=r.brand or "unknown"))

        for c in cues:
            res = ground_cue(c, r, ctx)
            verdict = "N/A (dropped)" if res is None else f"score={res.score} via {res.tool}"
            print(f"      ground {c.type.value}={c.value!r:30} -> {verdict}")
        print()

    print("Phase 2 tools OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
