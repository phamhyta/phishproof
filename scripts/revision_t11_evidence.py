#!/usr/bin/env -S uv run --quiet
"""T11 — verifier provenance, cue dependence, and controlled attack evidence
.

Subcommands (all offline; nothing here calls a model API):

  verifiers        Re-run the four verifier soundness evaluations with ONE RECORD PER
                   SAMPLED UNIT (page x tested claim), explicit label provenance
                   (dataset gold / strict password-field gold / parser-derived), and
                   confusion matrices reconstructed from those records. Also
                   reconciles the committed artifacts/verifier_soundness.json against
                   the historical 998-page split it came from.

  dependence       Same-value coincidence between agent pairs under a VALUE-SPECIFIC
                   permutation null on the common assertion population, with absolute
                   counts, per cue type; false values split into gold-annotated-false
                   (brand/logo vs dataset brand) and tool-refuted (parser-derived).
                   The larger panel is all-cross-family, so no same-family effect is
                   claimed from it (stated in the output).

  false-acceptance Conditional error rates the propositions actually need: among
                   FALSE CONSENSUS-SELECTED cues (strict-majority set from the T8
                   eligibility records), how often does the verifier accept? Wilson
                   intervals; natural panel errors only (attack strata come from
                   attack-policy; random negatives live in `verifiers`).

  attack-policy    Rebuild every attacked page (same deterministic perturbations and
                   capture files as run_adversarial_or.py), replay the panel from the
                   cache, apply the FROZEN complete policy of T8 to the clean and the
                   attacked capture of the SAME page, and emit paired records plus the
                   strict gate-attribution partition (score_abstain / veto_only /
                   wrong_act, summing exactly to the evaded count) with Wilson CIs,
                   plus the matched clean-control abstention at the same threshold.

Usage
    uv run scripts/revision_t11_evidence.py dependence
    uv run scripts/revision_t11_evidence.py verifiers
    uv run scripts/revision_t11_evidence.py false-acceptance
    uv run scripts/revision_t11_evidence.py attack-policy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling script imports

import numpy as np

OUT_DIR = Path("results/revision_v2/t11")
T8_DIR = Path("results/revision_v2/t8")

BUNDLES = {
    "phishpedia": "results/revision/bundle_or_full.jsonl",
    "apwg": "results/revision/bundle_apwg_or_full.jsonl",
    "trop": "results/revision/bundle_trop_or_full.jsonl",
}
MANIFESTS = {
    "phishpedia": "data/phishsel_final/test.jsonl",
    "apwg": "data/apwg_final/test.jsonl",
    "trop": "data/trop_final/test.jsonl",
}


def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta(extra: dict | None = None) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    d = {"git_sha": sha, "command": " ".join(sys.argv),
         "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if extra:
        d.update(extra)
    return d


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, float(c - h)), min(1.0, float(c + h))


def load_rows(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


# ===================================================================== dependence
def _norm(ctype: str, value: str) -> str | None:
    from phishproof.aggregate.normalize import normalize_value
    from phishproof.schema import CueType
    return normalize_value(CueType(ctype), value)


def dependence(args) -> int:
    from phishproof.tools.brands import canonical_brand

    rng = np.random.RandomState(args.seed)
    out: dict = {"meta": meta({"seed": args.seed, "n_perm": args.n_perm}), "corpora": {}}
    per_records: list[dict] = []

    for corpus, bpath in BUNDLES.items():
        rows = [r for r in load_rows(bpath) if r.get("agents")]
        gold_by_pid = {}
        for m in load_rows(MANIFESTS[corpus]):
            if m.get("brand"):
                gold_by_pid[m["page_id"]] = canonical_brand(m["brand"])

        agents = sorted({a["id"] for r in rows for a in r["agents"]})
        pairs = [(a, b) for i, a in enumerate(agents) for b in agents[i + 1:]]
        ctypes = sorted({c["type"] for r in rows for a in r["agents"]
                         for c in a.get("cues", [])})

        # per page x agent x type -> set of normalized values
        vals: dict[str, dict[str, dict[str, set]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
        for r in rows:
            for a in r["agents"]:
                for c in a.get("cues", []):
                    v = _norm(c["type"], c["value"])
                    if v:
                        vals[r["page_id"]][a["id"]][c["type"]].add(v)

        cres = {}
        for a, b in pairs:
            for t in ctypes:
                pop = [pid for pid in vals
                       if vals[pid][a].get(t) and vals[pid][b].get(t)]
                n = len(pop)
                if n == 0:
                    cres[f"{a}|{b}|{t}"] = {"n_common": 0}
                    continue
                A = [vals[pid][a][t] for pid in pop]
                B = [vals[pid][b][t] for pid in pop]
                same = sum(1 for x, y in zip(A, B) if x & y)

                # value-specific permutation null: shuffle B across pages, keeping each
                # model's own per-page value sets (marginals preserved exactly)
                null_counts = np.empty(args.n_perm)
                for i in range(args.n_perm):
                    perm = rng.permutation(n)
                    null_counts[i] = sum(1 for j in range(n) if A[j] & B[perm[j]])
                null_mean = float(null_counts.mean())
                p_perm = float((null_counts >= same).mean())

                # false-value analysis where an independent gold exists (brand/logo)
                rec = {"n_common": n, "same_value": same,
                       "null_mean": round(null_mean, 2),
                       "null_sd": round(float(null_counts.std()), 2),
                       "ratio_obs_over_null": round(same / null_mean, 2)
                       if null_mean > 0 else None,
                       "p_perm_ge": round(p_perm, 4)}
                if t in ("brand_claim", "logo_brand"):
                    gpop = [j for j, pid in enumerate(pop) if pid in gold_by_pid]
                    both_false = same_false = 0
                    for j in gpop:
                        g = gold_by_pid[pop[j]]
                        fa = {v for v in A[j] if v != g}
                        fb = {v for v in B[j] if v != g}
                        if fa and fb:
                            both_false += 1
                            if fa & fb:
                                same_false += 1
                    rec["gold_annotated"] = {
                        "n_with_gold": len(gpop), "both_assert_false": both_false,
                        "same_false_value": same_false,
                        "provenance": "dataset brand annotation (independent of tools)",
                    }
                cres[f"{a}|{b}|{t}"] = rec
                per_records.append({"corpus": corpus, "pair": f"{a}|{b}",
                                    "type": t, **rec})
        out["corpora"][corpus] = {
            "bundle": bpath, "n_rows": len(rows), "agents": agents,
            "pairs": cres,
        }

    out["family_note"] = (
        "The larger panel is all-cross-family (Meta Llama-3.3-70B, Alibaba Qwen-2.5-72B, "
        "OpenAI GPT-4o) and all pairs are cross-family; a same-family effect is NOT "
        "identifiable from it. A matched same-family comparison needs the "
        "panel_samefamily configuration (3B panel) and is reported separately if run.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cue_dependence.json").write_text(json.dumps(out, indent=2))
    with (OUT_DIR / "cue_dependence_records.jsonl").open("w") as fh:
        for r in per_records:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({c: {k: v for k, v in d["pairs"].items() if v.get("n_common", 0) > 0}
                      for c, d in out["corpora"].items()}, indent=2)[:4000])
    print(f"[ok] wrote {OUT_DIR}/cue_dependence.json")
    return 0


# ===================================================================== verifiers
def verifiers(args) -> int:
    from phishproof.data_io import read_manifest
    from phishproof.schema import Cue, CueType
    from phishproof.tools.brands import canonical_brand
    from phishproof.tools.detector import HtmlBrandDetector
    from phishproof.tools.dom import (
        form_action_domains,
        has_credential_intent,
    )
    from phishproof.tools.logo_brand import CLIPLogoEmbedder, crop_logo
    from eval_verifier_soundness import strict_password_field

    t_logo = None
    pol = T8_DIR / "policy_frozen.json"
    if pol.exists():
        t_logo = json.loads(pol.read_text())["t_logo"]

    rng = random.Random(args.seed)
    records: list[dict] = []

    def add(rec):
        records.append(rec)

    datasets = {"phishpedia_4020": "data/phishsel_final",
                "historical_998": "data/phishsel_final.998.bak"}
    summary: dict = {"meta": meta({"seed": args.seed, "t_logo_frozen": t_logo,
                                   "legacy_logo_threshold": 0.5})}

    for dsname, root in datasets.items():
        pages = read_manifest(Path(root) / "test.jsonl")
        all_brands = sorted({canonical_brand(p.brand) for p in pages
                             if canonical_brand(p.brand)})
        all_domains = sorted({d for p in pages for d in form_action_domains(p)})
        det = HtmlBrandDetector()

        # ---- brand_claim: gold-positive + random-negative units ------------------
        bc = Counter()
        for p in pages:
            b = canonical_brand(p.brand)
            if not b:
                continue
            s = det.grounds(b, p)
            add({"dataset": dsname, "cue": "brand_claim", "page_id": p.page_id,
                 "unit": "gold_positive", "tested_value": b, "score": s,
                 "label_provenance": "dataset brand annotation",
                 "accepted": s >= 1.0})
            bc["pos_total"] += 1
            bc["pos_accept"] += int(s >= 1.0)
            others = [o for o in all_brands if o != b]
            if others:
                neg = rng.choice(others)
                s2 = det.grounds(neg, p)
                add({"dataset": dsname, "cue": "brand_claim", "page_id": p.page_id,
                     "unit": "random_negative", "tested_value": neg, "score": s2,
                     "label_provenance": "random other dataset brand "
                                         "(unaimed negative, NOT adaptive)",
                     "accepted": s2 >= 1.0})
                bc["neg_total"] += 1
                bc["neg_accept"] += int(s2 >= 1.0)

        # ---- form_action_domain: parser-derived positive + fabricated negative --
        fa = Counter()
        for p in pages:
            doms = form_action_domains(p)
            for d in doms:
                add({"dataset": dsname, "cue": "form_action_domain",
                     "page_id": p.page_id, "unit": "parser_positive",
                     "tested_value": d, "score": 1.0,
                     "label_provenance": "PARSER-DERIVED (same DOM parser as the "
                                         "verifier; cannot validate parser "
                                         "correctness)",
                     "accepted": True})
                fa["pos_total"] += 1
                fa["pos_accept"] += 1
            if doms:
                fake = [d for d in all_domains if d not in doms]
                if fake:
                    neg = rng.choice(fake)
                    from phishproof.tools.dom import verify_form_action_domain
                    s2 = verify_form_action_domain(
                        Cue(type=CueType.FORM_ACTION_DOMAIN, value=neg), p)
                    add({"dataset": dsname, "cue": "form_action_domain",
                         "page_id": p.page_id, "unit": "random_negative",
                         "tested_value": neg, "score": s2,
                         "label_provenance": "fabricated domain from corpus pool",
                         "accepted": s2 >= 1.0})
                    fa["neg_total"] += 1
                    fa["neg_accept"] += int(s2 >= 1.0)

        # ---- credential_intent: strict password-field gold ----------------------
        cr = Counter()
        for p in pages:
            gold = strict_password_field(p)
            pred = has_credential_intent(p)
            add({"dataset": dsname, "cue": "credential_intent", "page_id": p.page_id,
                 "unit": "page", "tested_value": "yes" if pred else "no",
                 "gold": bool(gold),
                 "label_provenance": "strict <input type=password> gold "
                                     "(independent of the hint-based predictor)",
                 "accepted": bool(pred)})
            cr["tp"] += int(pred and gold)
            cr["fp"] += int(pred and not gold)
            cr["fn"] += int(not pred and gold)
            cr["tn"] += int(not pred and not gold)

        # ---- logo_brand: CLIP on subsample, both thresholds ----------------------
        lg = Counter()
        emb = CLIPLogoEmbedder()
        cand = [p for p in pages if canonical_brand(p.brand)
                and crop_logo(p) is not None]
        rng.shuffle(cand)
        cand = cand[: args.logo_limit]
        for p in cand:
            b = canonical_brand(p.brand)
            crop = crop_logo(p)
            s = float(emb.similarity(crop, b))
            others = [o for o in all_brands if o != b]
            neg = rng.choice(others)
            s2 = float(emb.similarity(crop, neg))
            for unit, val, sc in (("gold_positive", b, s), ("random_negative", neg, s2)):
                add({"dataset": dsname, "cue": "logo_brand", "page_id": p.page_id,
                     "unit": unit, "tested_value": val, "score": sc,
                     "label_provenance": "dataset brand annotation"
                     if unit == "gold_positive" else "random other dataset brand",
                     "accepted_legacy_0.5": sc > 0.5,
                     "accepted_frozen": (sc >= t_logo) if t_logo else None})
            lg["n"] += 1
            lg["pos_legacy"] += int(s > 0.5)
            lg["neg_legacy_rej"] += int(s2 <= 0.5)
            if t_logo:
                lg["pos_frozen"] += int(s >= t_logo)
                lg["neg_frozen_rej"] += int(s2 < t_logo)

        summary[dsname] = {
            "manifest": str(Path(root) / "test.jsonl"),
            "manifest_sha256": sha256_file(Path(root) / "test.jsonl"),
            "n_pages": len(pages),
            "brand_claim": {"recall": bc["pos_accept"] / bc["pos_total"],
                            "specificity": 1 - bc["neg_accept"] / bc["neg_total"],
                            "n_pos": bc["pos_total"], "n_neg": bc["neg_total"]},
            "form_action_domain": {
                "recall_parser_derived": fa["pos_accept"] / fa["pos_total"]
                if fa["pos_total"] else None,
                "specificity": 1 - fa["neg_accept"] / fa["neg_total"]
                if fa["neg_total"] else None,
                "n_pos": fa["pos_total"], "n_neg": fa["neg_total"],
                "caveat": "positives are parser-derived; recall=1 is circular by "
                          "construction and cannot validate the parser"},
            "credential_intent": {
                "confusion": dict(cr),
                "n": sum(cr.values()),
                "precision": cr["tp"] / (cr["tp"] + cr["fp"])
                if (cr["tp"] + cr["fp"]) else None,
                "recall": cr["tp"] / (cr["tp"] + cr["fn"])
                if (cr["tp"] + cr["fn"]) else None,
                "specificity": cr["tn"] / (cr["tn"] + cr["fp"])
                if (cr["tn"] + cr["fp"]) else None},
            "logo_brand": {
                "n": lg["n"],
                "recall_legacy_0.5": lg["pos_legacy"] / lg["n"] if lg["n"] else None,
                "specificity_legacy_0.5": lg["neg_legacy_rej"] / lg["n"]
                if lg["n"] else None,
                "recall_frozen": lg["pos_frozen"] / lg["n"]
                if lg["n"] and t_logo else None,
                "specificity_frozen": lg["neg_frozen_rej"] / lg["n"]
                if lg["n"] and t_logo else None,
                "t_logo_frozen": t_logo},
        }
        print(f"[{dsname}] done", flush=True)

    # ---- reconciliation of the committed artifact -------------------------------
    committed = json.loads(Path("artifacts/verifier_soundness.json").read_text())
    cred_row = next(r for r in committed if r["cue"] == "credential_intent")
    h = summary["historical_998"]["credential_intent"]["confusion"]
    summary["reconciliation"] = {
        "committed_file": "artifacts/verifier_soundness.json",
        "committed_credential_note": cred_row.get("note"),
        "recomputed_998_confusion": h,
        "match": (f"tp={h['tp']} fp={h['fp']} fn={h['fn']} tn={h['tn']}"
                  in (cred_row.get("note") or "")),
        "explanation": "the committed artifact was produced on the historical 998-page "
                       "test split (499 phish + 499 benign); its credential n=998 is "
                       "that split's page count, not a subsample of the 4,020 split. "
                       "The current-split records above are the authoritative ones.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "verifier_unit_records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "verifier_soundness_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"},
                     indent=2, default=str)[:3000])
    print(f"[ok] wrote {OUT_DIR}/verifier_soundness_v2.json "
          f"({len(records)} unit records)")
    return 0


# ===================================================================== false-acceptance
def false_acceptance(args) -> int:
    from phishproof.tools.brands import canonical_brand

    out: dict = {"meta": meta(), "corpora": {}}
    for corpus in BUNDLES:
        elig = T8_DIR / f"eligibility_{corpus}.jsonl"
        if not elig.exists():
            print(f"[skip] {elig} missing — run T8 first")
            continue
        rows = load_rows(elig)
        gold_by_pid = {m["page_id"]: canonical_brand(m["brand"])
                       for m in load_rows(MANIFESTS[corpus]) if m.get("brand")}

        per_type = defaultdict(Counter)
        recs = []
        for r in rows:
            gold = gold_by_pid.get(r["page_id"])
            for c in r.get("cues", []):
                if not c["in_strict_consensus"]:
                    continue
                t = c["type"]
                per_type[t]["consensus_cues"] += 1
                false_known = None
                provenance = None
                if t in ("brand_claim", "logo_brand") and gold:
                    false_known = c["value"] != gold
                    provenance = "dataset brand annotation"
                elif t in ("form_action_domain", "credential_intent"):
                    # tool-refuted only: the verifier itself is the label source
                    false_known = (c["available"] and c["raw_score"] == 0.0)
                    provenance = "TOOL-REFUTED (parser-derived; not independent gold)"
                if false_known is None:
                    per_type[t]["no_label"] += 1
                    continue
                if false_known:
                    per_type[t]["false_cues"] += 1
                    accepted = bool(c["passed"])
                    per_type[t]["false_accepted"] += int(accepted)
                    recs.append({"corpus": corpus, "page_id": r["page_id"],
                                 "type": t, "value": c["value"], "gold": gold,
                                 "score": c["raw_score"], "passed": c["passed"],
                                 "provenance": provenance})
                else:
                    per_type[t]["true_cues"] += 1

        cres = {}
        for t, c in per_type.items():
            k, n = c["false_accepted"], c["false_cues"]
            lo, hi = wilson(k, n)
            cres[t] = {**dict(c),
                       "false_acceptance_rate": round(k / n, 4) if n else None,
                       "wilson_95": [round(lo, 4), round(hi, 4)] if n else None,
                       "stratum": "natural panel errors on clean pages "
                                  "(NOT adaptive; see attack-policy for attacks)"}
        out["corpora"][corpus] = cres
        with (OUT_DIR / f"false_consensus_cues_{corpus}.jsonl").open("w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "false_acceptance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["corpora"], indent=2))
    print(f"[ok] wrote {OUT_DIR}/false_acceptance.json")
    return 0


# ===================================================================== attack-policy
def attack_policy(args) -> int:
    from phishproof.data_io import read_manifest
    from phishproof.eval.perturb import (
        cloak_form_action,
        occlude_logo,
        strip_brand_text,
    )
    from revision_t8_policy import (
        Grounder,
        RawCacheReader,
        load_calibrator,
        replay_page,
    )
    from phishproof.config import load_panel

    ATTACKS = {
        "cloak": [cloak_form_action],
        "occlude": [occlude_logo],
        "both": [cloak_form_action, occlude_logo],
        "adaptive": [cloak_form_action, strip_brand_text, occlude_logo],
    }

    policy = json.loads((T8_DIR / "policy_frozen.json").read_text())
    t_logo, tau = policy["t_logo"], policy["tau"]
    cx, cy, _sha = load_calibrator(Path("results/calibrator_or.json"))
    panel_cfg = load_panel(Path("configs/panel_or.yaml"))
    cfgs = panel_cfg.panel
    reader = RawCacheReader()
    grounder = Grounder(use_logo=True)

    # the historical attacked sample: exactly the pages in the recorded bundle
    hist = load_rows("results/_or/bundle_adversarial_or.jsonl")
    sample = defaultdict(list)
    for r in hist:
        sample[r["attack"]].append(r["page_id"])

    pages = {p.page_id: p for p in read_manifest(Path("data/phishsel_final/test.jsonl"))}
    clean_elig = {r["page_id"]: r
                  for r in load_rows(T8_DIR / "eligibility_phishpedia.jsonl")}

    out_rows = []
    summary = {}
    for attack, pids in sorted(sample.items()):
        fns = ATTACKS[attack]
        n_missing_panel = 0
        arows = []
        t0 = time.time()
        for i, pid in enumerate(pids, 1):
            p = pages[pid]
            ap = p
            for fn in fns:
                ap = fn(ap, Path("results/perturbed_or"))
            if ap.page_id == pid:
                continue
            rec = replay_page(ap, cfgs, reader, grounder, t_logo, tau, cx, cy)
            miss = any(not v["raw_present"] for v in rec["agents"])
            n_missing_panel += int(miss)
            clean = clean_elig.get(pid)
            row = {
                "page_id": pid, "attack": attack,
                "original_correct": clean is not None
                and clean["verdict"] == "phish",
                "attacked_verdict": rec["verdict"],
                "evaded": rec["verdict"] != "phish",
                "inference_valid_attacked": rec["complete"]["valid_inference"],
                "trust_attacked": rec["trust_calibrated"],
                "trust_with_fallback_attacked": rec["trust_with_fallback"],
                "clean_action": clean["complete"]["action"] if clean else None,
                "clean_primary_reason": clean["complete"]["primary_reason"]
                if clean else None,
                "attacked_action": rec["complete"]["action"],
                "attacked_primary_reason": rec["complete"]["primary_reason"],
                "attacked_score_pass": rec["complete"]["score_pass"],
                "attacked_all_pass": rec["complete"]["all_pass"],
                "attacked_all_available": rec["complete"]["all_available"],
                "attacked_n_consensus": rec["complete"]["n_consensus"],
                "cue_availability": [
                    {"type": c["type"], "value": c["value"],
                     "in_consensus": c["in_strict_consensus"],
                     "available": c["available"], "passed": c["passed"],
                     "score": c["raw_score"]} for c in rec["cues"]],
                "replay_missing_agent": miss,
            }
            out_rows.append(row)
            arows.append(row)
            if i % 25 == 0:
                print(f"  {attack} {i}/{len(pids)} ({time.time()-t0:.0f}s)", flush=True)

        evaded = [r for r in arows if r["evaded"]]
        # STRICT ORDERED partition of the evaded pages (must sum exactly). Order matters:
        # an invalid inference can never act regardless of the score/verification gates,
        # so it is its OWN bucket, checked before veto_only. Without this, an evaded page
        # that abstains because one agent's JSON was malformed would be miscounted as a
        # verification veto even though every consensus check passed.
        invalid_inference = [r for r in evaded if not r["inference_valid_attacked"]]
        rest = [r for r in evaded if r["inference_valid_attacked"]]
        score_abstain = [r for r in rest if not r["attacked_score_pass"]]
        veto_only = [r for r in rest if r["attacked_score_pass"]
                     and r["attacked_action"] == "abstain"]
        wrong_act = [r for r in rest if r["attacked_action"] == "act"]
        assert (len(invalid_inference) + len(score_abstain) + len(veto_only)
                + len(wrong_act) == len(evaded))
        ne = len(evaded)
        k_ii = len(invalid_inference)
        k_sa, k_vo, k_wa = len(score_abstain), len(veto_only), len(wrong_act)
        non_evaded = [r for r in arows if not r["evaded"]]
        clean_ctrl = [clean_elig[r["page_id"]] for r in arows
                      if r["page_id"] in clean_elig]
        clean_abstain = sum(1 for c in clean_ctrl
                            if c["complete"]["action"] == "abstain")
        summary[attack] = {
            "n_attacked": len(arows),
            "n_missing_agent_on_replay": n_missing_panel,
            "evaded_n": ne,
            "evade_pct": round(100 * ne / len(arows), 1) if arows else None,
            "partition": {
                "invalid_inference": k_ii, "score_abstain": k_sa,
                "veto_only": k_vo, "wrong_act": k_wa,
                "sums_to_evaded": k_ii + k_sa + k_vo + k_wa == ne,
                "invalid_inference_wilson": [round(x, 3) for x in wilson(k_ii, ne)]
                if ne else None,
                "score_abstain_wilson": [round(x, 3) for x in wilson(k_sa, ne)]
                if ne else None,
                "veto_only_wilson": [round(x, 3) for x in wilson(k_vo, ne)]
                if ne else None,
                "wrong_act_wilson": [round(x, 3) for x in wilson(k_wa, ne)]
                if ne else None,
                "note": "invalid_inference = evaded page that also fails validity "
                        "(cannot act at any threshold); score_abstain = valid but "
                        "kappa(s) < tau; veto_only = score accepted, verification gate "
                        "rejected; wrong_act = passed both gates and acted.",
            },
            "non_evaded_stratum": {
                "n": len(non_evaded),
                "abstain": sum(1 for r in non_evaded
                               if r["attacked_action"] == "abstain"),
                "act": sum(1 for r in non_evaded if r["attacked_action"] == "act"),
            },
            "matched_clean_control": {
                "n": len(clean_ctrl),
                "abstain": clean_abstain,
                "abstain_pct": round(100 * clean_abstain / len(clean_ctrl), 1)
                if clean_ctrl else None,
                "note": "same pages, clean capture, SAME complete policy and tau "
                        "(full-policy control, not a score-only rate)",
            },
        }
        print(f"[{attack}] {json.dumps(summary[attack]['partition'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "attack_policy_records.jsonl").open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "attack_policy_summary.json").write_text(json.dumps(
        {"meta": meta({"policy_frozen": str(T8_DIR / "policy_frozen.json"),
                       "t_logo": t_logo, "tau": tau,
                       "sample": "the recorded adversarial sample "
                                 "(results/_or/bundle_adversarial_or.jsonl); "
                                 "expansion is a separate budgeted run"}),
         "attacks": summary}, indent=2))
    print(f"[ok] wrote {OUT_DIR}/attack_policy_records.jsonl + summary")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dependence")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--n-perm", type=int, default=1000)
    v = sub.add_parser("verifiers")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--logo-limit", type=int, default=250)
    sub.add_parser("false-acceptance")
    sub.add_parser("attack-policy")
    args = ap.parse_args()
    return {"dependence": dependence, "verifiers": verifiers,
            "false-acceptance": false_acceptance,
            "attack-policy": attack_policy}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
