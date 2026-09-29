#!/usr/bin/env -S uv run --quiet
"""T8 — implement and freeze COMPLETE verification.

Replays the larger panel from the response cache (ZERO API calls; a cache miss is a
recorded model/API failure, never a network request) and, for every manifest row, exports
the full eligibility/action record of the frozen complete policy in
phishproof/aggregate/policy.py:

    ACT iff valid inference AND kappa(s) >= tau AND strict-majority consensus nonempty
        AND every consensus check available AND every consensus check passes.

Two modes:

  --fit-logo-threshold : replay the CALIBRATION manifest, score every cited logo cue with
      CLIP, label it against the dataset's independent gold-brand annotation, select the
      perceptual pass threshold (max Youden J), and FREEZE it (with the whole rule set)
      into results/revision_v2/t8/policy_frozen.json. Run this once, before any test use.

  default : replay a TEST manifest under the frozen policy; write per-row eligibility
      records, the old-surrogate vs complete-gate transition table with reason counts,
      and the missing-artifact inventory. Replayed gea/verdict/groundedness are validated
      against the historical replay bundle; the script refuses to write on a mismatch.

Usage
    uv run scripts/revision_t8_policy.py --fit-logo-threshold
    uv run scripts/revision_t8_policy.py --corpus phishpedia
    uv run scripts/revision_t8_policy.py --corpus apwg
    uv run scripts/revision_t8_policy.py --corpus trop
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phishproof.agents.base import _parse as _deployed_parse
from phishproof.agents.page_context import render_context
from phishproof.agents.panel import Panel
from phishproof.agents.prompts import SYSTEM_PROMPT, build_user_prompt
from phishproof.aggregate.consensus import consensus_cues, per_type_agreement
from phishproof.aggregate.policy import (
    POLICY_VERSION,
    AgentValidity,
    Reason,
    build_cue_checks,
    evaluate,
)
from phishproof.cache import JsonCache
from phishproof.config import load_panel
from phishproof.data_io import read_manifest
from phishproof.schema import AgentOutput, Cue, CueType, Label, PageRecord
from phishproof.tools.consistency import build_consistency_cue
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder, crop_logo
from phishproof.tools.registry import GroundingContext, ground_cue

EPS = 1e-3
VISION_ID = "agent_c_vision"

CORPORA = {
    "phishpedia": dict(manifest="data/phishsel_final/test.jsonl",
                       validate="results/revision/bundle_or_full.jsonl"),
    "apwg": dict(manifest="data/apwg_final/test.jsonl",
                 validate="results/revision/bundle_apwg_or_full.jsonl"),
    "trop": dict(manifest="data/trop_final/test.jsonl",
                 validate="results/revision/bundle_trop_or_full.jsonl"),
    "calibration": dict(manifest="data/phishsel_final/calibration.jsonl", validate=None),
    # the SMALLER (3B) panel on the same Phishpedia test rows -- bundle_D has no
    # per-agent cues, so the T10 smaller-panel ladder needs this replay. Validation
    # compares gea+verdict only (bundle_D stores no per-run groundedness convention
    # marker); the panel config differs.
    "phishpedia_smaller": dict(manifest="data/phishsel_final/test.jsonl",
                               validate="results/bundle_D.jsonl",
                               panel="configs/panel.yaml",
                               validate_fields=("gea", "verdict")),
}

OUT_DIR = Path("results/revision_v2/t8")
POLICY_PATH = OUT_DIR / "policy_frozen.json"


# ------------------------------------------------------------------ run metadata
def sha256_file(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_meta(args: argparse.Namespace, panel_cfg) -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    return {
        "policy_version": POLICY_VERSION,
        "git_sha": sha,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "cache-only (no network; a miss is a recorded failure)",
        "panel_config": str(args.panel),
        "panel_config_sha256": sha256_file(args.panel),
        "models": {a.id: a.model for a in panel_cfg.panel},
        "prompt_schema": "phishproof.agents.prompts.RawAgentResponse "
                         "(SYSTEM_PROMPT sha256 %s)" % hashlib.sha256(
                             SYSTEM_PROMPT.encode()).hexdigest()[:16],
        "clip_model": "ViT-B-32-quickgelu/openai (open_clip)",
        "seed": "deterministic (temperature 0 at capture time; replay is exact)",
        "retries": "n/a on replay; original-run failures appear as cache misses",
    }


# ------------------------------------------------------------------ cache replay
class RawCacheReader:
    """Read the RAW stored model response for one agent+page (never the network)."""

    def __init__(self) -> None:
        self.cache = JsonCache()

    def raw(self, cfg, page: PageRecord) -> str | None:
        user = build_user_prompt(render_context(page))
        image = page.screenshot_path if cfg.modality == "vision" else None
        img_hash = JsonCache.hash_image(image) if image else None
        prompt = f"SYS:{SYSTEM_PROMPT}\nUSR:{user}\nDETAIL:{cfg.detail}"
        return self.cache.get(cfg.model, prompt, img_hash)


_VALID_CUE_TYPES = {t.value for t in CueType}


def strict_validity(cfg, raw: str | None) -> tuple[AgentOutput, AgentValidity, int]:
    """Verdict-level (lenient-cue) parse + validity of one raw response.

    Design decision (Cach B, chosen by the author 2026-09-28): an agent's inference is
    VALID when its response yields a parseable verdict (and, for the vision agent, a
    numeric confidence). An off-schema CUE TYPE does not invalidate the verdict -- it is a
    mislabelled piece of evidence, so the cue is DROPPED (never counted in consensus) but
    the verdict and confidence are kept. This both matches the plan's intent
    ("...parses into schema-valid JSON with a verdict...") and fixes the audit's
    'parse failure silently becomes a benign label' bug: llama-3.2-3B, which labels every
    cue with an off-schema type, previously had its real verdict thrown away and replaced
    by a benign fallback; here its verdict is kept and only its cues are dropped.

    Returns (AgentOutput built from the lenient parse, validity, n_off_schema_cues_dropped).
    A missing raw response is a model/API failure (raw_present=False).
    """
    is_vision = cfg.modality == "vision"
    if raw is None:
        out = AgentOutput(agent_id=cfg.id, verdict=Label.BENIGN, cues=[])
        return out, AgentValidity(cfg.id, cfg.model, is_vision, raw_present=False), 0

    text = raw.strip()
    if not text.startswith("{"):
        i, j = text.find("{"), text.rfind("}")
        text = text[i: j + 1] if i != -1 and j != -1 else "{}"
    verdict = None
    conf = None
    cues: list[Cue] = []
    n_dropped = 0
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001 - unparseable JSON => no verdict => invalid inference
        data = None
    if isinstance(data, dict):
        v = str(data.get("verdict", "")).strip().lower()
        if v in ("phish", "benign"):
            verdict = Label(v)
        c = data.get("confidence", None)
        if isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0:
            conf = float(c)
        for raw_cue in data.get("cues", []) or []:
            if not isinstance(raw_cue, dict):
                continue
            ctype = str(raw_cue.get("type", "")).strip()
            cval = raw_cue.get("value", "")
            if ctype in _VALID_CUE_TYPES and str(cval).strip():
                cues.append(Cue(type=CueType(ctype), value=str(cval),
                                raw_value=str(cval), asserted_by=cfg.id))
            else:
                n_dropped += 1

    valid = verdict is not None
    has_conf = valid and conf is not None
    out = AgentOutput(agent_id=cfg.id, verdict=verdict or Label.BENIGN, cues=cues,
                      confidence=conf, raw_response=raw)
    return (out,
            AgentValidity(cfg.id, cfg.model, is_vision, raw_present=True,
                          valid_json=valid, has_confidence=has_conf, n_cues=len(cues)),
            n_dropped)


# ------------------------------------------------------------------ grounding
class Grounder:
    """Grounds candidate cues once per (page, type, value); reuses the logo crop."""

    LOGO_TOOL = "logo.clip ViT-B-32-quickgelu/openai"
    TOOLS = {
        "brand_claim": "detector.brand HtmlBrandDetector",
        "form_action_domain": "dom.form_action (bs4+lxml)",
        "credential_intent": "dom.credential_intent (bs4+lxml)",
        "brand_domain_consistency": "consistency.brand_domain",
    }

    def __init__(self, use_logo: bool = True) -> None:
        self.ctx = GroundingContext(detector=HtmlBrandDetector(), logo_embedder=None)
        self.embedder = CLIPLogoEmbedder() if use_logo else None
        if self.embedder:
            self.embedder._ensure()

    def ground(self, check, page: PageRecord, crop) -> None:
        """Attach (tool, raw_score, unavailable_reason) to one CueCheck, then finalize."""
        ctype = CueType(check.type)
        if ctype is CueType.LOGO_BRAND:
            check.tool = self.LOGO_TOOL
            if not page.screenshot_path or not Path(page.screenshot_path).exists():
                check.unavailable_reason = "missing_screenshot"
            elif crop is None:
                check.unavailable_reason = "no_logo_box"
            elif self.embedder is None:
                check.unavailable_reason = "clip_unavailable"
            else:
                check.raw_score = float(self.embedder.similarity(crop, check.value))
        else:
            check.tool = self.TOOLS.get(check.type, check.type)
            cue = Cue(type=ctype, value=check.value,
                      raw_value=(check.raw_values[0] if check.raw_values else None))
            r = ground_cue(cue, page, self.ctx)
            if r is None:
                check.unavailable_reason = "tool_na"
            else:
                check.raw_score = float(r.score)
        check.finalize()


# ------------------------------------------------------------------ calibration map
def load_calibrator(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    d = json.loads(path.read_text())
    assert d["kind"] == "isotonic"
    return np.asarray(d["x"], float), np.asarray(d["y"], float), sha256_file(path)


def kappa(s: float, cx: np.ndarray, cy: np.ndarray) -> float:
    return float(np.interp(s, cx, cy))


# ------------------------------------------------------------------ page pipeline
def replay_page(page: PageRecord, cfgs, reader: RawCacheReader, grounder: Grounder,
                t_logo: float, tau: float, cx, cy) -> dict:
    captures = {
        "screenshot": bool(page.screenshot_path and Path(page.screenshot_path).exists()),
        "html": bool(page.dom_html_path and Path(page.dom_html_path).exists()),
    }
    outs: list[AgentOutput] = []
    vals: list[AgentValidity] = []
    deployed_outs: list[AgentOutput] = []   # what the ORIGINAL pipeline recorded
    n_dropped_by_agent: dict[str, int] = {}
    for cfg in cfgs:
        raw = reader.raw(cfg, page)
        out, v, n_drop = strict_validity(cfg, raw)
        outs.append(out)
        vals.append(v)
        n_dropped_by_agent[cfg.id] = n_drop
        deployed_outs.append(
            _deployed_parse(cfg.id, raw) if raw is not None
            else AgentOutput(agent_id=cfg.id, verdict=Label.BENIGN, cues=[]))

    verdict = Panel.majority_label(outs)
    gea, _ = per_type_agreement(outs)
    # the deployed-parse panel result reproduces the historical bundle exactly; used only
    # to validate that the replay is faithful (the lenient parse is the authoritative one)
    deployed_verdict = Panel.majority_label(deployed_outs).value
    deployed_gea, _ = per_type_agreement(deployed_outs)
    vis = next((o for o in outs if o.agent_id == VISION_ID), None)
    conf = vis.confidence if vis is not None else None
    # deployed convention (run_adversarial_or.formula_trust): the reference verdict for
    # a_text is the vision agent's whenever a vision output exists
    ref = vis.verdict if vis is not None else verdict
    text_outs = [o for o in outs if o.agent_id != VISION_ID]
    a_text = (sum(1 for o in text_outs if o.verdict == ref) / len(text_outs)
              if text_outs else 0.0)

    # --- OLD surrogate quantities (exactly the recorded evaluation's convention) ------
    cons_relaxed = consensus_cues(outs, relax_perceptual=True)
    brand = next((c.value for c in cons_relaxed if c.type is CueType.BRAND_CLAIM), None)
    derived = build_consistency_cue(brand, page) if brand else None
    from phishproof.aggregate.consensus import shared_cue_set
    pool = shared_cue_set(outs)
    if derived is not None:
        cons_old = list(cons_relaxed) + [derived]
        agreement_old = len(cons_old) / (len(pool) + 1) if (len(pool) + 1) else 0.0
    else:
        cons_old = list(cons_relaxed)
        agreement_old = (len(cons_old) / len(pool)) if pool else 0.0

    # --- one grounding pass over the strict candidate pool + old-consensus extras -----
    checks = build_cue_checks(outs, t_logo, derived_cue=derived)
    crop = crop_logo(page) if any(c.type == "logo_brand" for c in checks) else None
    scores_by_key: dict[tuple[str, str], float | None] = {}
    for c in checks:
        grounder.ground(c, page, crop)
        scores_by_key[(c.type, c.value)] = c.raw_score
    # the relaxed consensus is a subset of pool + derived, so every member is scored
    g_scores = [scores_by_key.get((c.type.value, c.value)) for c in cons_old]
    g_avail = [s for s in g_scores if s is not None]
    g_old = sum(g_avail) / len(g_avail) if g_avail else 0.0

    s_fb = (conf if conf is not None else 0.5) + EPS * a_text   # deployed 0.5 fallback
    trust_fb = kappa(s_fb, cx, cy)
    s_strict = None if conf is None else conf + EPS * a_text
    trust_strict = None if s_strict is None else kappa(s_strict, cx, cy)

    act_old = bool(trust_fb >= tau and len(cons_old) > 0 and g_old >= 0.999)

    decision = evaluate(vals, captures, checks, trust_strict, tau)

    return {
        "page_id": page.page_id,
        "label": page.label.value,
        "is_hard": (page.source or "").endswith("hard"),
        "verdict": verdict.value if verdict else None,
        "gea": gea,
        "agents": [dataclasses.asdict(v) for v in vals],
        "conf_vlm": conf,
        "a_text": a_text,
        "off_schema_cues_dropped": n_dropped_by_agent,
        # per-agent verdict + confidence under the SAME lenient parse, so T10 can
        # recompute B1/B2/B4/B6 consistently (not from the bundle's fallback parse)
        "agent_outputs": [{"id": o.agent_id, "verdict": o.verdict.value,
                           "confidence": o.confidence} for o in outs],
        "_deployed_verdict": deployed_verdict,
        "_deployed_gea": deployed_gea,
        "s_raw": s_strict,
        "s_with_fallback": s_fb,
        "trust_calibrated": trust_strict,
        "trust_with_fallback": trust_fb,
        "surrogate": {
            "n_consensus_relaxed": len(cons_old),
            "agreement": agreement_old,
            "groundedness_mean": g_old,
            "score_pass": bool(trust_fb >= tau),
            "act": act_old,
        },
        "complete": dataclasses.asdict(decision),
        "cues": [dataclasses.asdict(c) for c in checks],
    }


# ------------------------------------------------------------------ threshold fitting
def fit_logo_threshold(args, cfgs, reader, grounder, meta) -> None:
    pages = read_manifest(Path(CORPORA["calibration"]["manifest"]))
    if args.limit:
        pages = pages[: args.limit]
    from phishproof.tools.brands import canonical_brand

    recs, n_no_gold, t0 = [], 0, time.time()
    for i, p in enumerate(pages, 1):
        outs = []
        for cfg in cfgs:
            raw = reader.raw(cfg, p)
            out, _v, _nd = strict_validity(cfg, raw)
            outs.append(out)
        gold = canonical_brand(p.brand) if p.brand else None
        vals = {}
        for o in outs:
            for c in o.cues:
                if c.type is CueType.LOGO_BRAND:
                    from phishproof.aggregate.normalize import normalize_cue
                    nc = normalize_cue(c)
                    if nc is not None:
                        vals.setdefault(nc.value, set()).add(o.agent_id)
        if not vals:
            continue
        if gold is None:
            n_no_gold += 1
            continue
        crop = crop_logo(p)
        for v, agents in vals.items():
            if crop is None:
                continue
            score = float(grounder.embedder.similarity(crop, v))
            recs.append({"page_id": p.page_id, "value": v, "n_asserting": len(agents),
                         "score": score, "true_claim": bool(canonical_brand(v) == gold)})
        if i % 100 == 0:
            print(f"  calib {i}/{len(pages)}  {time.time()-t0:.0f}s", flush=True)

    y = np.array([r["true_claim"] for r in recs], bool)
    s = np.array([r["score"] for r in recs], float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    grid = np.unique(np.round(s, 4))
    best = None
    curve = []
    for t in grid:
        tpr = float((s[y] >= t).mean()) if n_pos else 0.0
        fpr = float((s[~y] >= t).mean()) if n_neg else 0.0
        curve.append({"t": float(t), "tpr": round(tpr, 4), "fpr": round(fpr, 4)})
        j = tpr - fpr
        if best is None or j > best[0]:
            best = (j, float(t), tpr, fpr)
    spec95 = next((c for c in curve if c["fpr"] <= 0.05), None)

    policy = {
        "meta": meta,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "consensus": "strict majority > M/2 on the same normalized (type, value); "
                         "no perceptual relaxation; derived cues are never members",
            "validity": "every agent: stored raw response parseable as RawAgentResponse "
                        "with a verdict; vision agent additionally a numeric confidence "
                        "in [0,1]; both captures present. Violations abstain permanently.",
            "structural_pass": "score >= 1.0 (deterministic 0/1 tools)",
            "perceptual_pass": "CLIP similarity >= t_logo (frozen below)",
            "unavailable_check": "never passes; forces abstention",
            "score_gate": "kappa(s) >= tau, s = conf_VLM + 1e-3 * a_text (eq. 7)",
        },
        "t_logo": best[1],
        "t_logo_selection": {
            "protocol": "max Youden J on independent calibration annotations "
                        "(cue value vs dataset gold brand), frozen before test use",
            "manifest": CORPORA["calibration"]["manifest"],
            "manifest_sha256": sha256_file(CORPORA["calibration"]["manifest"]),
            "n_cues": len(recs), "n_true": n_pos, "n_false": n_neg,
            "n_pages_with_logo_no_gold_excluded": n_no_gold,
            "youden_j": round(best[0], 4), "tpr_at_t": round(best[2], 4),
            "fpr_at_t": round(best[3], 4),
            "alternative_spec95": spec95,
            "note": "gold brand is the dataset's page-level brand annotation, available "
                    "for the phishing half of the calibration split; benign pages carry "
                    "no gold brand and are excluded from the fit (counted above).",
        },
        "tau": args.tau,
        "calibrator": str(args.calibrator),
        "calibrator_sha256": sha256_file(args.calibrator),
        "roc_curve": curve,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(json.dumps(policy, indent=2))
    (OUT_DIR / "logo_calibration_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"[frozen] t_logo={best[1]:.4f} (J={best[0]:.3f}, tpr={best[2]:.3f}, "
          f"fpr={best[3]:.3f}; {n_pos} true / {n_neg} false cues) -> {POLICY_PATH}")


# ------------------------------------------------------------------ corpus run
def run_corpus(args, cfgs, reader, grounder, meta) -> int:
    spec = CORPORA[args.corpus]
    policy = json.loads(POLICY_PATH.read_text())
    t_logo, tau = policy["t_logo"], policy["tau"]
    cx, cy, cal_sha = load_calibrator(Path(args.calibrator))
    assert cal_sha == policy["calibrator_sha256"], "calibrator changed after freeze"

    pages = read_manifest(Path(spec["manifest"]))
    if args.limit:
        pages = pages[: args.limit]

    rows, t0 = [], time.time()
    for i, p in enumerate(pages, 1):
        rows.append(replay_page(p, cfgs, reader, grounder, t_logo, tau, cx, cy))
        if i % 200 == 0:
            el = time.time() - t0
            print(f"  {args.corpus} {i}/{len(pages)}  {el:.0f}s "
                  f"(~{el/i*(len(pages)-i)/60:.0f} min left)", flush=True)

    # ---- validation against the historical replay bundle -------------------------
    # Two distinct parses are involved. The DEPLOYED parse (phishproof.agents.base._parse,
    # the one that produced the bundle) must reproduce the bundle exactly -- that is the
    # faithfulness-of-replay check, and any failure there is a real replay bug. The
    # verdict-level (lenient) parse is the NEW authoritative one; wherever it disagrees
    # with the bundle it is CORRECTING a deployed parse-fallback (llama emitted a valid
    # verdict that RawAgentResponse rejected wholesale, so the bundle recorded a benign
    # fallback -- sometimes mislabelling a true phish page as benign). Those corrections
    # are counted and reported, not treated as failures.
    validation = {"validated_against": spec["validate"], "compared": 0,
                  "skipped_replay_failed": 0,
                  "deployed_verdict_mismatch": 0, "deployed_gea_mismatch": 0,
                  "lenient_corrections_verdict": 0, "lenient_corrections_gea": 0,
                  "corrections_that_fix_label": 0}
    if spec["validate"]:
        old = {json.loads(l)["page_id"]: json.loads(l)
               for l in Path(spec["validate"]).read_text().splitlines() if l.strip()}
        vfields = spec.get("validate_fields", ("gea", "verdict", "groundedness"))
        for r in rows:
            o = old.get(r["page_id"])
            if o is None:
                continue
            if o.get("replay_failed"):
                validation["skipped_replay_failed"] += 1
                continue
            validation["compared"] += 1
            # (1) faithfulness: DEPLOYED parse must reproduce the bundle exactly
            if "verdict" in vfields and o["verdict"] != r["_deployed_verdict"]:
                validation["deployed_verdict_mismatch"] += 1
            if "gea" in vfields and abs(float(o["gea"]) - float(r["_deployed_gea"])) > 1e-9:
                validation["deployed_gea_mismatch"] += 1
            # (2) Cach B corrections: lenient parse differs from the bundle
            if "verdict" in vfields and o["verdict"] != r["verdict"]:
                validation["lenient_corrections_verdict"] += 1
                if r["verdict"] == r["label"] and o["verdict"] != o.get("label"):
                    validation["corrections_that_fix_label"] += 1
            if "gea" in vfields and abs(float(o["gea"]) - float(r["gea"])) > 1e-9:
                validation["lenient_corrections_gea"] += 1
        bad = (validation["deployed_verdict_mismatch"]
               + validation["deployed_gea_mismatch"])
        print(f"[validate] {validation}")
        if bad:
            print("[validate] FAIL — the DEPLOYED parse does not reproduce the bundle "
                  "(real replay bug); refusing to write outputs")
            return 2
        print("[validate] OK — deployed parse reproduces the bundle exactly; the lenient "
              f"parse makes {validation['lenient_corrections_verdict']} verdict + "
              f"{validation['lenient_corrections_gea']} gea corrections "
              f"({validation['corrections_that_fix_label']} fix a mislabelled page).")

    # ---- transition table + reason counts ----------------------------------------
    from collections import Counter
    trans = Counter()
    reasons = Counter()
    for r in rows:
        old_a = "act" if r["surrogate"]["act"] else "abstain"
        new_a = r["complete"]["action"]
        trans[f"{old_a}->{new_a}"] += 1
        reasons[r["complete"]["primary_reason"]] += 1
    missing = {
        "cache_misses_by_agent": dict(Counter(
            v["agent_id"] for r in rows for v in r["agents"] if not v["raw_present"])),
        "pages_missing_screenshot": sum(
            1 for r in rows if "invalid:missing_capture:screenshot"
            in r["complete"]["invalid_reasons"]),
        "pages_missing_html": sum(
            1 for r in rows if "invalid:missing_capture:html"
            in r["complete"]["invalid_reasons"]),
        "note": "these are the artifacts a re-run would need; nothing was fetched",
    }
    summary = {
        "meta": meta,
        "corpus": args.corpus,
        "manifest": spec["manifest"],
        "manifest_sha256": sha256_file(spec["manifest"]),
        "n_rows": len(rows),
        "policy_frozen": str(POLICY_PATH), "t_logo": t_logo, "tau": tau,
        "validation": validation,
        "transition_old_surrogate_vs_complete": dict(trans),
        "complete_primary_reason_counts": dict(reasons),
        "old_surrogate_acts": sum(1 for r in rows if r["surrogate"]["act"]),
        "complete_acts": sum(1 for r in rows if r["complete"]["action"] == "act"),
        "missing_artifacts": missing,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT_DIR / f"eligibility_{args.corpus}.jsonl"
    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / f"transition_{args.corpus}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("meta",)}, indent=2))
    print(f"[ok] wrote {out_jsonl} and transition_{args.corpus}.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=list(CORPORA), default="phishpedia")
    ap.add_argument("--panel", type=Path, default=Path("configs/panel_or.yaml"))
    ap.add_argument("--calibrator", type=Path, default=Path("results/calibrator_or.json"))
    ap.add_argument("--tau", type=float, default=0.9651,
                    help="operating point on kappa(s); the code value (text rounds to .965)")
    ap.add_argument("--fit-logo-threshold", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-logo", action="store_true",
                    help="skip CLIP; logo checks become unavailable (smoke tests only)")
    args = ap.parse_args()

    if not args.fit_logo_threshold:
        cpanel = CORPORA[args.corpus].get("panel")
        if cpanel:
            args.panel = Path(cpanel)
    panel_cfg = load_panel(args.panel)
    cfgs = panel_cfg.panel
    reader = RawCacheReader()
    grounder = Grounder(use_logo=not args.no_logo)
    meta = run_meta(args, panel_cfg)

    if args.fit_logo_threshold:
        if grounder.embedder is None:
            print("[err] --fit-logo-threshold needs CLIP (drop --no-logo)")
            return 1
        fit_logo_threshold(args, cfgs, reader, grounder, meta)
        return 0
    if not POLICY_PATH.exists():
        print(f"[err] {POLICY_PATH} not found — run --fit-logo-threshold first "
              "(the threshold must be frozen before test use)")
        return 1
    return run_corpus(args, cfgs, reader, grounder, meta)


if __name__ == "__main__":
    raise SystemExit(main())
