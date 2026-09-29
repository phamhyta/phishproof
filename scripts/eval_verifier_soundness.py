"""RO-4: per-cue-type verifier soundness (P0 from the five-element review).

The paper reported precision/recall only for the brand and logo verifiers. This measures
ALL FOUR active verifiers as detectors of an independently-derivable ground-truth property,
so a reviewer cannot say "only one verifier was validated".

Per cue type, the verifier should fire iff its target property holds:
  - brand_claim (HtmlBrandDetector): page presents brand B. Gold = dataset brand annotation.
        recall = grounds(gold_brand);  specificity = 1 - grounds(random_other_brand)
  - logo_brand (CLIP, sim>0.5):       page shows logo of brand B. Gold = brand + screenshot.
        recall = sim(gold_brand)>0.5;  specificity = 1 - sim(random_other_brand)>0.5
  - form_action_domain (DOM):         a form posts to domain D. Gold = DOM-parsed actions.
        recall = verify(real_action_domain)->1; specificity = 1 - verify(fabricated_domain)
  - credential_intent (DOM):          page collects credentials. Strict gold = <input
        type=password> present (independent of the hint-based predictor).
        precision/recall of has_credential_intent vs the strict password-field gold.

Subsamples the (slow) CLIP logo check; the deterministic verifiers run on the full split.

Usage: .venv/bin/python scripts/eval_verifier_soundness.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from phishproof.data_io import read_manifest
from phishproof.schema import Cue, CueType, PageRecord
from phishproof.tools.brands import canonical_brand
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.dom import (
    form_action_domains,
    has_credential_intent,
    verify_credential_intent,
    verify_form_action_domain,
)
from phishproof.tools.logo_brand import CLIPLogoEmbedder, crop_logo


def _pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return p, r


def strict_password_field(page: PageRecord) -> bool:
    if not page.dom_html_path or not Path(page.dom_html_path).exists():
        return False
    soup = BeautifulSoup(Path(page.dom_html_path).read_text(errors="replace"), "lxml")
    return any((i.get("type") or "").lower() == "password" for i in soup.find_all("input"))


def eval_brand_claim(pages, all_brands, rng):
    rec_hit = rec_tot = spec_hit = spec_tot = 0
    det = HtmlBrandDetector()
    for p in pages:
        b = canonical_brand(p.brand)
        if not b:
            continue
        rec_tot += 1
        if det.grounds(b, p) >= 1.0:
            rec_hit += 1
        others = [o for o in all_brands if o != b]
        if others:
            spec_tot += 1
            if det.grounds(rng.choice(others), p) < 1.0:
                spec_hit += 1
    return {
        "cue": "brand_claim", "tool": "HtmlBrandDetector",
        "recall": rec_hit / rec_tot if rec_tot else float("nan"),
        "specificity": spec_hit / spec_tot if spec_tot else float("nan"),
        "n": rec_tot,
    }


def eval_form_action(pages, all_domains, rng):
    rec_hit = rec_tot = spec_hit = spec_tot = 0
    for p in pages:
        doms = form_action_domains(p)
        for d in doms:
            rec_tot += 1
            if verify_form_action_domain(Cue(type=CueType.FORM_ACTION_DOMAIN, value=d), p) >= 1.0:
                rec_hit += 1
        if doms:
            fake = [d for d in all_domains if d not in doms]
            if fake:
                spec_tot += 1
                c = Cue(type=CueType.FORM_ACTION_DOMAIN, value=rng.choice(fake))
                if verify_form_action_domain(c, p) < 1.0:
                    spec_hit += 1
    return {
        "cue": "form_action_domain", "tool": "dom.form_action",
        "recall": rec_hit / rec_tot if rec_tot else float("nan"),
        "specificity": spec_hit / spec_tot if spec_tot else float("nan"),
        "n": rec_tot,
    }


def eval_credential(pages):
    tp = fp = fn = tn = 0
    for p in pages:
        gold = strict_password_field(p)
        pred = has_credential_intent(p)
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    p_, r_ = _pr(tp, fp, fn)
    return {
        "cue": "credential_intent", "tool": "dom.credential_intent",
        "precision": p_, "recall": r_,
        "note": f"vs strict <input type=password> gold (tp={tp} fp={fp} fn={fn} tn={tn})",
        "n": tp + fp + fn + tn,
    }


def eval_logo(pages, all_brands, rng, limit, thr=0.5):
    emb = CLIPLogoEmbedder()
    cand = [p for p in pages if canonical_brand(p.brand) and crop_logo(p) is not None]
    rng.shuffle(cand)
    cand = cand[:limit]
    rec_hit = spec_hit = 0
    for i, p in enumerate(cand, 1):
        b = canonical_brand(p.brand)
        crop = crop_logo(p)
        if emb.similarity(crop, b) > thr:
            rec_hit += 1
        others = [o for o in all_brands if o != b]
        if emb.similarity(crop, rng.choice(others)) <= thr:
            spec_hit += 1
        if i % 50 == 0:
            print(f"  logo {i}/{len(cand)}", flush=True)
    n = len(cand)
    return {
        "cue": "logo_brand", "tool": "CLIP ViT-B-32",
        "recall": rec_hit / n if n else float("nan"),
        "specificity": spec_hit / n if n else float("nan"),
        "n": n, "threshold": thr,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--logo-limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pages = read_manifest(args.data / "test.jsonl")
    all_brands = sorted({canonical_brand(p.brand) for p in pages if canonical_brand(p.brand)})
    all_domains = sorted({d for p in pages for d in form_action_domains(p)})

    rows = []
    print("brand_claim...");        rows.append(eval_brand_claim(pages, all_brands, rng))
    print("form_action_domain..."); rows.append(eval_form_action(pages, all_domains, rng))
    print("credential_intent...");  rows.append(eval_credential(pages))
    print("logo_brand (CLIP, subsampled)..."); rows.append(eval_logo(pages, all_brands, rng, args.logo_limit))

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "  -"

    print(f"\n{'cue':22}{'tool':22}{'precision':>10}{'recall':>9}{'spec':>8}{'n':>7}")
    for r in rows:
        print(f"{r['cue']:22}{r['tool']:22}"
              f"{fmt(r.get('precision')):>10}"
              f"{fmt(r.get('recall')):>9}"
              f"{fmt(r.get('specificity')):>8}"
              f"{r['n']:>7}")

    Path("results/verifier_soundness.json").write_text(json.dumps(rows, indent=2))
    print("\n[ok] wrote results/verifier_soundness.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
