"""RO-6 / Upgrade C: Episodic Similarity Prior — ceiling probe for brand recovery.

Diagnosis (audit s1.3): 73/76 missed-phish FN have NO brand_claim in consensus — the small
agents fail to *name* the impersonated brand. ESP retrieves the k most DOM-similar pages
from the (labeled) calibration split and borrows their brand as a soft prior, gated by
grounding (the prior only counts if the test page actually presents that brand).

This probe measures the CEILING of that idea without changing the panel:
  1. TF-IDF over each page's DOM identity text; cosine k-NN test -> calibration.
  2. Candidate brand = most common gold brand among the k neighbors.
  3. Grounded prior = HtmlBrandDetector.grounds(candidate, test_page) (page must show it).
  4. FN recovery ceiling: of the false-negatives (phish called benign), how many get a
     grounded brand prior they were missing?
  5. Actionable rule: flip to phish if (grounded brand prior) AND (credential intent OR
     off-domain form action). Report FN recovered and NEW false positives created.

No model calls (cache-free): uses DOM HTML only.

Usage: .venv/bin/python scripts/run_esp.py --data data/phishsel_final
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from phishproof.data_io import read_manifest
from phishproof.eval.bundle import load_bundle
from phishproof.tools.brands import canonical_brand
from phishproof.tools.consistency import consistency_for_brand
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.dom import form_action_domains, has_credential_intent
from phishproof.tools.urls import registrable_domain


def identity_text(det: HtmlBrandDetector, page) -> str:
    return det._identity_text(page) or ""


def off_domain_form(page) -> bool:
    base = registrable_domain(page.final_url or page.url)
    doms = form_action_domains(page)
    return bool(doms) and any(d != base for d in doms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/phishsel_final"))
    ap.add_argument("--bundle", type=Path, default=Path("results/bundle_final.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sim-thresh", type=float, default=0.0,
                    help="only trust the prior if the top neighbor cosine sim >= this")
    args = ap.parse_args()

    det = HtmlBrandDetector()
    calib = read_manifest(args.data / "calibration.jsonl")
    test = read_manifest(args.data / "test.jsonl")
    test_by_id = {p.page_id: p for p in test}
    bundle = load_bundle(args.bundle)

    print("building DOM-identity TF-IDF features...")
    calib_txt = [identity_text(det, p) for p in calib]
    test_txt = [identity_text(det, p) for p in test]
    vec = TfidfVectorizer(min_df=2, max_features=20000, ngram_range=(1, 2))
    Xc = vec.fit_transform(calib_txt)
    Xt = vec.transform(test_txt)
    calib_brand = [canonical_brand(p.brand) for p in calib]

    print(f"retrieving k={args.k} neighbors per test page...")
    sims = cosine_similarity(Xt, Xc)  # (n_test, n_calib)
    topk = sims.argsort(axis=1)[:, ::-1][:, : args.k]

    prior_brand = {}
    top_sim = {}
    for i, p in enumerate(test):
        brands = [calib_brand[j] for j in topk[i] if calib_brand[j]]
        prior_brand[p.page_id] = Counter(brands).most_common(1)[0][0] if brands else None
        top_sim[p.page_id] = float(sims[i, topk[i][0]])

    # analyze on the bundle
    fn = []   # phish called benign
    fp = []   # benign called phish
    benign_correct = []
    for r in bundle:
        pid = r["page_id"]
        if r["label"] == "phish" and r["verdict"] == "benign":
            fn.append(pid)
        elif r["label"] == "benign" and r["verdict"] == "phish":
            fp.append(pid)
        elif r["label"] == "benign" and r["verdict"] == "benign":
            benign_correct.append(pid)

    cons_brand = {r["page_id"]: any(c["type"] == "brand_claim" for c in r.get("consensus_cues", []))
                  for r in bundle}

    def grounded_prior(pid):
        b = prior_brand.get(pid)
        if not b or top_sim.get(pid, 0.0) < args.sim_thresh:
            return False
        return det.grounds(b, test_by_id[pid]) >= 1.0

    # FN recovery ceiling
    fn_no_brand = [p for p in fn if not cons_brand.get(p, False)]
    fn_recovered = [p for p in fn_no_brand if grounded_prior(p)]

    # actionable rule on FN: grounded prior brand is shown but on an UNRELATED domain
    # (brand-domain inconsistency = the real phishing signal; legit pages are excluded
    # because their brand sits on its own domain).
    def rule_flip(pid):
        page = test_by_id[pid]
        b = prior_brand.get(pid)
        if not b or not grounded_prior(pid):
            return False
        return consistency_for_brand(b, page) == "inconsistent"

    fn_rule = [p for p in fn if rule_flip(p)]
    # new FP: benign-correct pages the rule would WRONGLY flip to phish
    fp_new = [p for p in benign_correct if rule_flip(p)]

    print(f"\ntest pages={len(bundle)}  FN(missed phish)={len(fn)}  FP={len(fp)}  "
          f"benign-correct={len(benign_correct)}")
    print(f"FN without brand in consensus: {len(fn_no_brand)}/{len(fn)}")
    print(f"  of those, grounded retrieved-brand prior available: {len(fn_recovered)} "
          f"(ceiling for ESP brand recovery)")
    print(f"\nactionable rule (grounded prior brand AND brand-domain inconsistent):")
    print(f"  FN flipped to phish (recovered): {len(fn_rule)}/{len(fn)}")
    print(f"  NEW false positives on benign-correct: {len(fp_new)}/{len(benign_correct)}")
    net = len(fn_rule) - len(fp_new)
    print(f"  net detection change: {net:+d} pages")

    out = {
        "k": args.k,
        "n_test": len(bundle),
        "fn": len(fn), "fp": len(fp), "benign_correct": len(benign_correct),
        "fn_no_brand": len(fn_no_brand),
        "fn_recovery_ceiling": len(fn_recovered),
        "rule_fn_recovered": len(fn_rule),
        "rule_new_fp": len(fp_new),
        "net_detection_change": net,
    }
    Path("results/esp.json").write_text(json.dumps(out, indent=2))
    print("\n[ok] wrote results/esp.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
