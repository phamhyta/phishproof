#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""T10 — recompute failure-inclusive metrics and controlled ablations
.

Everything starts from the FULL manifest row set of each corpus, joined by page id to
the T8 eligibility records (results/revision_v2/t8/) and the T9 baseline records
(results/revision_v2/t9/). Conventions, stated once and applied everywhere:

  * A page is ELIGIBLE for a score when its inference is valid for that score's
    mechanism (T8 validity; gated variants add their gate). Ineligible pages stay in
    every coverage denominator as forced abstentions and are NEVER ranked into an
    acting prefix — no fallback value fabricates an action.
  * AURC is reported over ATTAINABLE coverage: the mean of risk(k) for k = 1..K over
    the eligible prefix (K = #eligible), with coverage measured against ALL manifest
    rows (c_max = K/N). Nothing is integrated to coverage one. The historical
    all-rows-ranked value is reported separately as `aurc_fallback_convention` for the
    old-vs-corrected table.
  * SelAcc80 exists only when c_max >= 0.80; otherwise it is reported as
    "not attainable", never as a fallback-score value.
  * Ties: the deterministic rule orders tied scores by page_id (label-independent);
    the tie-robust value is the mean over 50 random tie orders.
  * Calibrated columns use cross-fitted isotonic maps with DUPLICATE-GROUP-DISJOINT
    folds where a group map exists (Phishpedia; t5_grouped_split.json), fitted on
    eligible rows only. Pooled cross-fitted scores can reorder observations across
    folds; that caveat is embedded in the output.
  * The paired bootstrap resamples the INDEPENDENT UNIT (duplicate group where known,
    else page), preserves method pairing, and CONDITIONS ON FROZEN MAPS (maps are not
    refitted per resample; stated in the output). delta = row_method - reference.
  * Ten-bin ECE: equal-width bins on [0,1]; population = eligible rows of that score
    (N stated); ineligible rows are excluded from ECE and counted in the reason table.

Usage
    uv run scripts/revision_t10_metrics.py --corpus phishpedia
    uv run scripts/revision_t10_metrics.py --corpus apwg
    uv run scripts/revision_t10_metrics.py --corpus trop
    uv run scripts/revision_t10_metrics.py --corpus phishpedia_smaller
    uv run scripts/revision_t10_metrics.py --diffs        # old-vs-corrected table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

T8 = Path("results/revision_v2/t8")
T9 = Path("results/revision_v2/t9")
OUT = Path("results/revision_v2/t10")

EPS = 1e-3
N_BOOT = 2000
N_FOLDS = 5
N_TIE = 50

BUNDLES = {
    "phishpedia": "results/revision/bundle_or_full.jsonl",
    "apwg": "results/revision/bundle_apwg_or_full.jsonl",
    "trop": "results/revision/bundle_trop_or_full.jsonl",
    "phishpedia_smaller": "results/bundle_D.jsonl",
}
MANIFESTS = {
    "phishpedia": "data/phishsel_final/test.jsonl",
    "apwg": "data/apwg_final/test.jsonl",
    "trop": "data/trop_final/test.jsonl",
    "phishpedia_smaller": "data/phishsel_final/test.jsonl",
}


def load_rows(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta(extra=None) -> dict:
    d = {"git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
         "command": " ".join(sys.argv),
         "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    if extra:
        d.update(extra)
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (max(0.0, float(c - h)), min(1.0, float(c + h)))


# ------------------------------------------------------------- eligible-only metrics
def _order(score, elig, pid_rank, jitter=None):
    """Rank ELIGIBLE rows by score desc; ties by jitter (random) or page-id rank."""
    idx = np.where(elig)[0]
    tie = jitter[idx] if jitter is not None else pid_rank[idx]
    return idx[np.lexsort((tie, -score[idx]))]


def selective_metrics(score, correct, elig, pid_rank, rng) -> dict:
    n = len(score)
    k_el = int(elig.sum())
    c_max = k_el / n
    if k_el == 0:
        return {"n": n, "n_eligible": 0, "max_attainable_coverage": 0.0,
                "AURC_attainable": None, "SelAcc80": "not attainable",
                "Cov99": 0.0, "ECE10": None}
    order = _order(score, elig, pid_rank)
    err = np.cumsum(1.0 - correct[order])
    ks = np.arange(1, k_el + 1)
    aurc_att = 100.0 * float(np.mean(err / ks))
    # tie-robust
    vals = []
    for _ in range(N_TIE):
        o = _order(score, elig, pid_rank, jitter=rng.permutation(n).astype(float))
        e = np.cumsum(1.0 - correct[o])
        vals.append(100.0 * float(np.mean(e / ks)))
    # historical fallback convention: EVERY row ranked (ineligible last by -inf)
    s_fb = np.where(elig, score, -np.inf)
    o_fb = np.lexsort((pid_rank, -s_fb))
    e_fb = np.cumsum(1.0 - correct[o_fb])
    aurc_fb = 100.0 * float(np.mean(e_fb / np.arange(1, n + 1)))

    k80 = int(round(0.80 * n))
    sel80 = (100.0 * float(np.mean(correct[order[:k80]]))
             if c_max >= 0.80 else "not attainable")
    running = np.cumsum(correct[order]) / ks
    ok = np.where(running >= 0.99)[0]
    cov99 = float(ok.max() + 1) / n if len(ok) else 0.0

    p = np.clip(score[elig], 0.0, 1.0)
    c = correct[elig]
    edges = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        hi = i == 9
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if hi else (p < edges[i + 1]))
        if m.sum():
            ece += m.sum() / len(p) * abs(c[m].mean() - p[m].mean())
    return {"n": n, "n_eligible": k_el,
            "max_attainable_coverage": round(c_max, 4),
            "AURC_attainable": round(aurc_att, 2),
            "AURC_attainable_tie_robust": round(float(np.mean(vals)), 2),
            "AURC_attainable_tie_sd": round(float(np.std(vals)), 3),
            "aurc_fallback_convention": round(aurc_fb, 2),
            "SelAcc80": sel80 if isinstance(sel80, str) else round(sel80, 1),
            "Cov99": round(cov99, 3),
            "ECE10": round(float(ece), 4),
            "ECE10_population": f"{k_el} eligible rows (ineligible excluded, "
                                f"counted in the reason table)"}


def make_folds(n, groups, seed=0):
    rng = np.random.RandomState(seed)
    if groups is None:
        return np.array_split(rng.permutation(n), N_FOLDS)
    uniq = np.array(sorted(set(groups.tolist())))
    assign = {g: i % N_FOLDS for i, g in enumerate(rng.permutation(uniq))}
    which = np.array([assign[g] for g in groups])
    return [np.where(which == i)[0] for i in range(N_FOLDS)]


def crossfit(score, correct, elig, folds):
    """Cross-fitted isotonic on ELIGIBLE rows only; ineligible stay NaN."""
    out = np.full(len(score), np.nan)
    for i in range(N_FOLDS):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i])
        tr = train[elig[train]]
        te = test[elig[test]]
        if len(tr) < 10 or len(te) == 0:
            out[te] = score[te]
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(score[tr], correct[tr])
        out[te] = ir.predict(score[te])
    return out


def _dawid_skene_transductive(outs_by_pid: dict, pids: list, n_iter: int = 50):
    """Transductive Dawid-Skene posterior confidence per page, fit on the given verdicts.

    outs_by_pid maps page_id -> list of {"id","verdict","confidence"} dicts.
    """
    lab = {"benign": 0, "phish": 1}
    workers = sorted({o["id"] for outs in outs_by_pid.values() for o in outs})
    widx = {w: i for i, w in enumerate(workers)}
    obs = [{widx[o["id"]]: lab[o["verdict"]] for o in outs_by_pid[p]}
           for p in pids]
    n_items, n_classes, n_workers = len(obs), 2, len(workers)
    if n_items == 0 or n_workers == 0:
        return np.full(len(pids), np.nan)
    T = np.full((n_items, n_classes), 0.5)
    for i, o in enumerate(obs):
        c = np.zeros(n_classes)
        for _w, l in o.items():
            c[l] += 1
        if c.sum():
            T[i] = c / c.sum()
    for _ in range(n_iter):
        prior = T.mean(0) + 1e-9
        prior /= prior.sum()
        conf = np.full((n_workers, n_classes, n_classes), 1e-9)
        for i, o in enumerate(obs):
            for w, l in o.items():
                conf[w, :, l] += T[i]
        conf /= conf.sum(axis=2, keepdims=True)
        newT = np.zeros_like(T)
        for i, o in enumerate(obs):
            logp = np.log(prior).copy()
            for w, l in o.items():
                logp += np.log(conf[w, :, l])
            p = np.exp(logp - logp.max())
            newT[i] = p / p.sum()
        T = newT
    return T.max(axis=1)


# ---------------------------------------------------------------- corpus assembly
def assemble(corpus: str):
    elig_rows = load_rows(T8 / f"eligibility_{corpus}.jsonl")
    by_pid = {r["page_id"]: r for r in elig_rows}
    n = len(elig_rows)
    pids = [r["page_id"] for r in elig_rows]
    pid_rank = np.argsort(np.argsort(np.array(pids))).astype(float)

    label = np.array([r["label"] == "phish" for r in elig_rows])
    verdict_phish = np.array([r["verdict"] == "phish" for r in elig_rows])
    correct = np.array([r["verdict"] == r["label"] if r["verdict"] else False
                        for r in elig_rows], float)

    valid = np.array([r["complete"]["valid_inference"] for r in elig_rows])
    conf = np.array([r["conf_vlm"] if r["conf_vlm"] is not None else np.nan
                     for r in elig_rows])
    a_text = np.array([r["a_text"] for r in elig_rows])
    gea = np.array([r["gea"] for r in elig_rows])
    s = np.where(np.isnan(conf), np.nan, conf + EPS * a_text)
    g_old = np.array([r["surrogate"]["groundedness_mean"] for r in elig_rows])
    sur_ver = np.array([r["surrogate"]["n_consensus_relaxed"] > 0
                        and r["surrogate"]["groundedness_mean"] >= 0.999
                        for r in elig_rows])
    comp_gate = np.array([r["complete"]["consensus_nonempty"]
                          and r["complete"]["all_available"]
                          and r["complete"]["all_pass"] for r in elig_rows])
    act_complete = np.array([r["complete"]["action"] == "act" for r in elig_rows])

    # Panel-derived baselines recomputed from the eligibility agent_outputs, i.e. under
    # the SAME verdict-level (lenient-cue) parse as PhishProof -- NOT from the bundle's
    # deployed fallback parse -- so the comparison is like-for-like (T9/T10). B1/B2 use
    # confidence (unchanged by the parse); B4/B6 use verdicts (which the parse corrects on
    # off-schema rows). Reimplemented inline (this script runs in an isolated uv env that
    # does not have the phishproof package) but identical to phishproof.baselines.
    outs_by_pid = {r["page_id"]: r.get("agent_outputs", []) for r in elig_rows}

    def _b1(outs):  # single-model confidence: first agent, 1.0 if it reports none
        if not outs:
            return 0.0
        c = outs[0].get("confidence")
        return float(c) if c is not None else 1.0

    def _b2(outs):  # mean verbalized confidence
        vals = [o["confidence"] for o in outs if o.get("confidence") is not None]
        return float(sum(vals) / len(vals)) if vals else 1.0

    def _b4(outs):  # majority-label share
        if not outs:
            return 0.0
        c = Counter(o["verdict"] for o in outs)
        return c.most_common(1)[0][1] / len(outs)

    def _b6(outs):  # MultiPhishGuard proxy: mean confidence of the majority-agreeing agents
        if not outs:
            return 0.0
        maj = Counter(o["verdict"] for o in outs).most_common(1)[0][0]
        agree = [o for o in outs if o["verdict"] == maj]
        confs = [o["confidence"] for o in agree if o.get("confidence") is not None]
        return float(sum(confs) / len(confs)) if confs else len(agree) / len(outs)

    base = {
        "B1": np.array([_b1(outs_by_pid[p]) for p in pids], float),
        "B2": np.array([_b2(outs_by_pid[p]) for p in pids], float),
        "B4": np.array([_b4(outs_by_pid[p]) for p in pids], float),
        "B6": np.array([_b6(outs_by_pid[p]) for p in pids], float),
    }

    # transductive Dawid-Skene EM on the lenient-parse test verdicts (same parse)
    ds_t = _dawid_skene_transductive(outs_by_pid, pids)
    # frozen DS is the calibration-fit variant disclosed in T9 (deployed-parse calib);
    # carried through only if present, and labelled as such in the ladder note.
    ds_f = np.full(n, np.nan)
    ds_path = T9 / f"dawid_skene_scores_{corpus}.json"
    if ds_path.exists():
        ds = json.loads(ds_path.read_text())
        for i, p in enumerate(pids):
            if p in ds:
                ds_f[i] = ds[p]["frozen"]

    cons_rel = np.full(n, np.nan)      # consolidator as reliability of panel verdict
    cons_conf = np.full(n, np.nan)     # consolidator own confidence
    cons_correct = np.full(n, np.nan)  # consolidator end-to-end correctness
    pid_ix = {p: i for i, p in enumerate(pids)}
    cpath = T9 / f"consolidator_{corpus}.jsonl"
    if cpath.exists():
        for r in load_rows(cpath):
            if r.get("status") != "ok" or not r.get("parse_ok"):
                continue
            i = pid_ix.get(r["page_id"])
            if i is None:
                continue
            cv, cc = r["verdict"], r["confidence"]
            pv = r["panel_verdict"]
            cons_conf[i] = cc
            cons_rel[i] = cc if cv == pv else 1.0 - cc
            cons_correct[i] = float(cv == by_pid[r["page_id"]]["label"])

    # per-agent verdicts/confidences under the SAME lenient parse (for the ablations)
    agent_verdicts: dict[str, list] = {}
    agent_confs: dict[str, list] = {}
    for r in elig_rows:
        for a in r.get("agent_outputs", []):
            agent_verdicts.setdefault(a["id"], [None] * n)
            agent_confs.setdefault(a["id"], [None] * n)
    for i, r in enumerate(elig_rows):
        for a in r.get("agent_outputs", []):
            agent_verdicts[a["id"]][i] = a.get("verdict")
            agent_confs[a["id"]][i] = a.get("confidence")

    return dict(rows=elig_rows, pids=pids, pid_rank=pid_rank, n=n, label=label,
                verdict_phish=verdict_phish, correct=correct, valid=valid, conf=conf,
                a_text=a_text, gea=gea, s=s, g_old=g_old, sur_ver=sur_ver,
                comp_gate=comp_gate, act_complete=act_complete, base=base,
                ds_t=ds_t, ds_f=ds_f, cons_rel=cons_rel, cons_conf=cons_conf,
                cons_correct=cons_correct, agent_verdicts=agent_verdicts,
                agent_confs=agent_confs)


def load_groups(corpus, pids):
    if corpus not in ("phishpedia", "phishpedia_smaller"):
        return None, "no duplicate-group map for this corpus; unit = page (stated)"
    g = json.loads(Path("results/revision/t5_grouped_split.json").read_text())
    m = g["test_group_map"]
    gid = {p: m.get(p, f"solo::{p}") for p in pids}
    uniq = {v: i for i, v in enumerate(sorted(set(gid.values())))}
    return np.array([uniq[gid[p]] for p in pids]), \
        "duplicate groups from t5_grouped_split.json (dhash near-dup + url/domain)"


def frozen_combined_ranker(corpus):
    """conf+GEA combination fitted on CALIBRATION rows only (larger panel), frozen."""
    if corpus not in ("phishpedia", "apwg", "trop"):
        return None
    cal = load_rows("results/bundle_calib_or.jsonl")
    X, y = [], []
    for r in cal:
        vis = next((a for a in r["agents"] if a["id"] == "agent_c_vision"), None)
        if vis is None or vis.get("confidence") is None:
            continue
        X.append([vis["confidence"], r["gea"]])
        y.append(int(r["verdict"] == r["label"]))
    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.array(X), np.array(y))
    return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "fit_on": "results/bundle_calib_or.jsonl (710 calibration rows; "
                      "combination rule chosen on calibration data only)",
            "model": lr}


# ---------------------------------------------------------------- per-corpus run
def run_corpus(corpus: str) -> dict:
    d = assemble(corpus)
    n = d["n"]
    rng = np.random.RandomState(0)
    groups, group_note = load_groups(corpus, d["pids"])
    folds = make_folds(n, groups, seed=0)

    # ---- 1. reason table -----------------------------------------------------------
    primary = Counter(r["complete"]["primary_reason"] for r in d["rows"])
    overlap = Counter(f for r in d["rows"] for f in r["complete"]["failed_reasons"])
    agent_fail = Counter()
    for r in d["rows"]:
        for v in r["agents"]:
            if not v["raw_present"]:
                agent_fail[f"model_call_missing:{v['agent_id']}"] += 1
            elif not v["valid_json"]:
                agent_fail[f"invalid_json:{v['agent_id']}"] += 1
            elif v["is_vision"] and not v["has_confidence"]:
                agent_fail[f"missing_confidence:{v['agent_id']}"] += 1
    reason_table = {
        "primary_mutually_exclusive": dict(primary),
        "overlapping_flags": dict(overlap),
        "per_agent_failure_states": dict(agent_fail),
        "missing_vision_confidence_rows": int(np.isnan(d["conf"]).sum()),
        "invalid_inference_rows": int((~d["valid"]).sum()),
    }

    # ---- 2. the ladder ---------------------------------------------------------------
    combined = frozen_combined_ranker(corpus)
    scores: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    def put(name, score, elig, note):
        scores[name] = (np.asarray(score, float), np.asarray(elig, bool), note)

    valid = d["valid"]
    has_s = valid & ~np.isnan(d["s"])
    put("conf_VLM", d["conf"], valid & ~np.isnan(d["conf"]),
        "vision confidence alone")
    put("concurrence", d["a_text"], valid, "text agreement with vision verdict")
    put("GEA", d["gea"], valid, "mean per-type agreement")
    put("s_eq7", d["s"], has_s, "deployed s = conf + 1e-3*a_text")
    put("s_gate_surrogate", d["s"], has_s & d["sur_ver"],
        "s among pages passing the OLD surrogate gate (relaxed consensus + "
        "mean G >= 0.999); others forced abstention")
    put("s_gate_complete", d["s"], has_s & d["comp_gate"],
        "s among pages passing the COMPLETE verification gate (T8); others "
        "forced abstention")
    put("GEAxG_diagnostic", d["gea"] * d["g_old"], valid,
        "diagnostic only; distinct from the combined ranker")
    if combined:
        z = combined["intercept"] + combined["coef"][0] * d["conf"] \
            + combined["coef"][1] * d["gea"]
        put("conf_plus_GEA_frozen", 1 / (1 + np.exp(-z)), has_s,
            "logistic(conf, GEA), coefficients frozen on calibration split")
    put("B1_single_conf", d["base"]["B1"], valid & ~np.isnan(d["base"]["B1"]),
        "single-model confidence")
    put("B2_mean_conf", d["base"]["B2"], valid & ~np.isnan(d["base"]["B2"]),
        "mean panel confidence")
    put("B3_majority_vote", d["base"]["B4"], valid & ~np.isnan(d["base"]["B4"]),
        "majority-vote share (recomputed from lenient-parse verdicts)")
    put("B4_dawid_skene_transductive", d["ds_t"], valid & ~np.isnan(d["ds_t"]),
        "DS EM fitted on the scored test rows (transductive; disclosed)")
    put("B4_dawid_skene_frozen", d["ds_f"], valid & ~np.isnan(d["ds_f"]),
        "DS EM fitted on calibration rows, frozen (T9 disclosure; calibration panel "
        "outputs use the deployed parse, so treat as a secondary variant)")
    put("B5P_majority_conf_proxy", d["base"]["B6"],
        valid & ~np.isnan(d["base"]["B6"]),
        "majority-confidence proxy (the mechanism previously mislabelled B5)")
    if not np.all(np.isnan(d["cons_rel"])):
        put("B5_consolidator_reliability", d["cons_rel"],
            valid & ~np.isnan(d["cons_rel"]),
            "ACTUAL consolidator confidence read as reliability of the fixed panel "
            "verdict (parse failures = forced abstention)")

    ladder = {}
    cal_cache = {}
    for name, (sc, el, note) in scores.items():
        raw = selective_metrics(sc, d["correct"], el, d["pid_rank"], rng)
        cal = crossfit(sc, d["correct"], el, folds)
        calm = selective_metrics(np.where(np.isnan(cal), 0.0, cal), d["correct"],
                                 el & ~np.isnan(cal), d["pid_rank"], rng)
        cal_cache[name] = (cal, el)
        ladder[name] = {"note": note, "raw": raw, "calibrated": calm}

    # consolidator end-to-end (its own predictions) — separate, as required
    if not np.all(np.isnan(d["cons_conf"])):
        el = valid & ~np.isnan(d["cons_conf"])
        e2e = selective_metrics(d["cons_conf"], np.nan_to_num(d["cons_correct"]),
                                el, d["pid_rank"], rng)
        ladder["B5_consolidator_end_to_end"] = {
            "note": "consolidator's OWN verdict + confidence (changes predictions; "
                    "reported separately from the reliability reading)",
            "raw": e2e,
            "prediction_agreement_with_panel": round(float(np.mean(
                (d["cons_correct"] == d["correct"])[el])), 4),
        }

    # ---- 3. paired bootstrap (frozen maps, group units) ------------------------------
    unit = groups if groups is not None else np.arange(n)
    uniq_units = np.unique(unit)
    unit_idx = {u: np.where(unit == u)[0] for u in uniq_units}
    resamples = []
    for _ in range(N_BOOT):
        us = rng.choice(uniq_units, len(uniq_units), replace=True)
        resamples.append(np.concatenate([unit_idx[u] for u in us]))

    def aurc_att_idx(sc, el, idx):
        e = el[idx]
        if not e.any():
            return np.nan
        s_i = sc[idx][e]
        c_i = d["correct"][idx][e]
        o = np.argsort(-s_i, kind="mergesort")
        err = np.cumsum(1.0 - c_i[o])
        return 100.0 * float(np.mean(err / np.arange(1, len(o) + 1)))

    def paired(name_a, name_b, calibrated):
        if calibrated:
            sa, ea = cal_cache[name_a][0], cal_cache[name_a][1]
            sb, eb = cal_cache[name_b][0], cal_cache[name_b][1]
            sa = np.where(np.isnan(sa), -np.inf, sa)
            sb = np.where(np.isnan(sb), -np.inf, sb)
        else:
            sa, ea, _ = scores[name_a]
            sb, eb, _ = scores[name_b]
        ds_ = np.array([aurc_att_idx(sa, ea, i) - aurc_att_idx(sb, eb, i)
                        for i in resamples])
        ds_ = ds_[~np.isnan(ds_)]
        if len(ds_) < 50:   # not enough overlapping eligible rows for a stable interval
            return {"delta_mean": None, "ci95": None, "p_one_sided_le0": None,
                    "n_valid_resamples": int(len(ds_)),
                    "sign": f"{name_a} minus {name_b} (negative favours {name_a})",
                    "note": "insufficient overlapping eligible rows on this panel"}
        return {"delta_mean": round(float(ds_.mean()), 2),
                "ci95": [round(float(np.percentile(ds_, 2.5)), 2),
                         round(float(np.percentile(ds_, 97.5)), 2)],
                "p_one_sided_le0": round(float((ds_ <= 0).mean()), 4),
                "n_valid_resamples": int(len(ds_)),
                "sign": f"{name_a} minus {name_b} (negative favours {name_a})"}

    refs = ["s_eq7", "GEA"]
    comparators = [k for k in scores if k not in refs]
    boots = {}
    for ref in refs:
        boots[f"vs_{ref}_raw"] = {c: paired(c, ref, False) for c in comparators}
        boots[f"vs_{ref}_calibrated"] = {c: paired(c, ref, True) for c in comparators}

    # ---- 4. complete-policy operating metrics ---------------------------------------
    act = d["act_complete"]
    lab = d["label"]
    pred = d["verdict_phish"]
    fp = int((act & pred & ~lab).sum())
    tp = int((act & pred & lab).sum())
    fn_act = int((act & ~pred & lab).sum())
    tn = int((act & ~pred & ~lab).sum())
    n_benign = int((~lab).sum())
    n_phish = int(lab.sum())
    tpr = tp / n_phish if n_phish else float("nan")
    _, fpr_hi = wilson(fp, n_benign)
    fn_lo, fn_hi = wilson(fn_act, n_phish)
    cov_p = float(act[lab].mean()) if n_phish else 0.0
    cov_b = float(act[~lab].mean()) if n_benign else 0.0
    policy = {
        "acts": int(act.sum()), "coverage_pct": round(100 * float(act.mean()), 2),
        "risk_on_acted_pct": round(100 * float(1 - d["correct"][act].mean()), 3)
        if act.sum() else None,
        "confusion_on_acted": {"tp": tp, "fp": fp, "fn": fn_act, "tn": tn},
        "false_block_rate_benign": {"k": fp, "n": n_benign,
                                    "wilson_hi_pct": round(100 * fpr_hi, 3)},
        "false_allow_rate_phish": {"k": fn_act, "n": n_phish,
                                   "wilson_95_pct": [round(100 * fn_lo, 3),
                                                     round(100 * fn_hi, 3)]},
        "bounds_note": "all interval bounds are MARGINAL (per quantity), not joint",
        "prevalence_adjusted": {},
    }
    for pv in (0.1, 0.01, 0.001):
        denom = pv * tpr + (1 - pv) * fpr_hi
        ppv_lb = (pv * tpr) / denom if denom > 0 else 1.0
        npv_num = (1 - pv) * (1 - fpr_hi)
        fnr_hi_frac = fn_hi
        npv_den = npv_num + pv * fnr_hi_frac
        policy["prevalence_adjusted"][str(pv)] = {
            "ppv_lower_bound_pct": round(100 * ppv_lb, 2),
            "npv_lower_bound_pct": round(100 * npv_num / npv_den, 3)
            if npv_den > 0 else None,
            "escalation_pct": round(100 * (1 - (pv * cov_p + (1 - pv) * cov_b)), 2),
            "zero_event_note": "fp=0 cells use the Wilson upper bound, "
                               "never an observed 0 rate",
        }

    # ---- 5. ablations (vision removal, M=1-3) ---------------------------------------
    ablations = {}
    if corpus in ("phishpedia", "phishpedia_smaller"):
        ablations = run_ablations(d, folds, rng)

    # ---- 6. temporal (phishpedia larger panel only) ----------------------------------
    temporal = None
    if corpus == "phishpedia":
        temporal = run_temporal(d)

    out = {
        "meta": meta({
            "corpus": corpus,
            "manifest": MANIFESTS[corpus],
            "manifest_sha256": sha256_file(MANIFESTS[corpus]),
            "eligibility": str(T8 / f"eligibility_{corpus}.jsonl"),
            "bundle": BUNDLES[corpus],
            "group_folds": group_note,
            "fold_seed": 0, "n_folds": N_FOLDS, "n_boot": N_BOOT,
            "bootstrap_units": "duplicate groups" if groups is not None else "pages",
            "bootstrap_maps": "FROZEN cross-fitted maps (intervals condition on "
                              "the fitted maps; maps are not refitted per resample)",
            "crossfit_caveat": "pooled cross-fitted values can reorder observations "
                               "across folds",
            "tie_rule": "deterministic by page_id; tie-robust = mean over "
                        f"{N_TIE} random tie orders",
            "combined_ranker": {k: v for k, v in (combined or {}).items()
                                if k != "model"} or None,
        }),
        "accuracy_pct": round(100 * float(d["correct"].mean()), 2),
        "reason_table": reason_table,
        "ladder": ladder,
        "paired_bootstrap": boots,
        "complete_policy": policy,
        "ablations": ablations,
        "temporal": temporal,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"t10_{corpus}.json").write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {OUT}/t10_{corpus}.json")
    return out


# ---------------------------------------------------------------- ablations
def run_ablations(d, folds, rng):
    """Vision removal + M=1..3: FIXED-PREDICTION and END-TO-END variants.

    End-to-end recomputes, per subset S: the majority verdict (tie -> phish, the
    deployed rule), validity (every subset agent valid + captures), per-type
    agreement over S's cited cues, s (when the vision agent is in S), the complete
    gate (strict majority WITHIN S, every member check available and passing, using
    the already-grounded per-cue scores), and a cross-fitted recalibration inside the
    variant. Fixed-prediction keeps the FULL panel's predictions and only swaps the
    ranking score. M=4-5 are not run and not claimed.
    """
    rows = d["rows"]
    n = d["n"]
    agents = list(d["agent_verdicts"])
    vision = "agent_c_vision"
    text_agents = [a for a in agents if a != vision]
    subsets = {"M1_vision_only": [vision],
               "M2_text_only(vision_removed)": text_agents,
               "M3_full": agents}
    for t in text_agents:
        subsets[f"M2_{t}+vision"] = [t, vision]

    cap_bad = np.array([any(x.startswith("invalid:missing_capture")
                            for x in r["complete"]["invalid_reasons"])
                        for r in rows])
    validity = {a: np.array([v["raw_present"] and v["valid_json"]
                             and (not v["is_vision"] or v["has_confidence"])
                             for r in rows
                             for v in [next(x for x in r["agents"]
                                            if x["agent_id"] == a)]])
                for a in agents}

    out = {}
    for name, S in subsets.items():
        Sset = set(S)
        m = len(S)
        v_ok = ~cap_bad.copy()
        for a in S:
            v_ok &= validity[a]

        verdict_s, gea_s, comp = [], [], []
        for i, r in enumerate(rows):
            votes = [d["agent_verdicts"][a][i] for a in S
                     if d["agent_verdicts"][a][i] is not None]
            nph = sum(1 for v in votes if v == "phish")
            nbe = len(votes) - nph
            verdict_s.append("phish" if nph >= nbe and votes else
                             ("benign" if votes else None))
            by_type: dict[str, dict[str, int]] = defaultdict(dict)
            nonempty, ok = False, True
            for c in r["cues"]:
                if c["source"] != "model":
                    continue
                k = len(Sset.intersection(c["asserted_by"]))
                if k:
                    by_type[c["type"]][c["value"]] = max(
                        by_type[c["type"]].get(c["value"], 0), k)
                if k > m / 2:
                    nonempty = True
                    if not (c["available"] and c["passed"]):
                        ok = False
            gea_s.append(float(np.mean(
                [max(by_type.get(t, {}).values(), default=0) / m
                 for t in ("brand_claim", "form_action_domain",
                           "credential_intent", "logo_brand")])))
            comp.append(nonempty and ok)

        gea_arr = np.array(gea_s)
        comp_arr = np.array(comp) & v_ok
        correct_e2e = np.array([verdict_s[i] == rows[i]["label"]
                                if verdict_s[i] else False
                                for i in range(n)], float)
        if vision in S:
            conf = d["conf"]
            texts = [a for a in S if a != vision]
            if texts:
                agree = np.mean([[1.0 if d["agent_verdicts"][a][i]
                                  == d["agent_verdicts"][vision][i] else 0.0
                                  for a in texts] for i in range(n)], axis=1)
            else:
                agree = np.zeros(n)
            score = np.where(np.isnan(conf), np.nan, conf + EPS * agree)
            score_name = "s_subset"
            el = v_ok & ~np.isnan(score)
        else:
            score = gea_arr
            score_name = "gea_subset"
            el = v_ok

        # end-to-end: subset predictions, cross-fit recalibrated inside the variant
        cal = crossfit(score, correct_e2e, el, folds)
        e2e_raw = selective_metrics(score, correct_e2e, el, d["pid_rank"], rng)
        e2e_cal = selective_metrics(np.where(np.isnan(cal), 0.0, cal), correct_e2e,
                                    el & ~np.isnan(cal), d["pid_rank"], rng)
        e2e_gate = selective_metrics(score, correct_e2e, el & comp_arr,
                                     d["pid_rank"], rng)
        # fixed-prediction: full-panel predictions, subset score as the ranker
        fixed = selective_metrics(score, d["correct"], el & d["valid"],
                                  d["pid_rank"], rng)
        out[name] = {
            "subset": S, "ranking_score": score_name,
            "n_valid_subset": int(v_ok.sum()),
            "accuracy_e2e_pct": round(100 * float(correct_e2e.mean()), 2),
            "end_to_end": {"raw": e2e_raw, "calibrated": e2e_cal,
                           "complete_gate": e2e_gate},
            "fixed_prediction": fixed,
            "call_count_per_page": len(S),
        }
    out["M4_M5"] = "not run; no artifacts exist and none are claimed"
    return out


# ---------------------------------------------------------------- temporal
def run_temporal(d):
    pat = re.compile(r"(20\d{2}-\d{2}-\d{2})")
    dates = {}
    for r in d["rows"]:
        m2 = pat.search(r["page_id"])
        if r["label"] == "phish" and m2:
            dates[r["page_id"]] = m2.group(1)
    if not dates:
        return {"status": "dates unavailable; limitation retained"}
    ds = sorted(dates.values())
    cutoff = ds[len(ds) // 2]
    early = {p for p, dt in dates.items() if dt <= cutoff}
    late = {p for p, dt in dates.items() if dt > cutoff}
    seg = {}
    for name, pool in (("early", early), ("late", late)):
        idx = [i for i, r in enumerate(d["rows"]) if r["page_id"] in pool]
        act = d["act_complete"][idx]
        cor = d["correct"][idx]
        k_act = int(act.sum())
        risk = float(1 - cor[act].mean()) if k_act else None
        lo, hi = wilson(int((1 - cor[act]).sum()), k_act) if k_act else (None, None)
        brands = Counter(r["page_id"].split("-2")[0] for r in
                         (d["rows"][i] for i in idx))
        seg[name] = {"n_phish": len(idx), "n_brands_approx": len(brands),
                     "acts": k_act,
                     "coverage_pct": round(100 * float(act.mean()), 1),
                     "risk_on_acted_pct": round(100 * risk, 2)
                     if risk is not None else None,
                     "risk_wilson_pct": [round(100 * lo, 2), round(100 * hi, 2)]
                     if k_act else None}
    return {
        "status": "PARTIAL phish-side diagnostic only",
        "cutoff_date": cutoff,
        "frozen": "deployed calibrator (fit on the calibration split) + tau; "
                  "no refit on either period",
        "segments": seg,
        "limitation_retained": "benign captures carry NO dates, so a fully "
                               "temporally disjoint calibration/test experiment is "
                               "impossible on this corpus; the benign side is "
                               "atemporal and shared across periods. Corpus identity "
                               "is NOT presented as time.",
    }


# ---------------------------------------------------------------- old-vs-new diffs
def run_diffs():
    pairs = {
        "phishpedia": ("results/revision/t1_larger_panel.json", "t10_phishpedia.json"),
        "apwg": ("results/revision/t1_apwg_larger.json", "t10_apwg.json"),
        "trop": ("results/revision/t1_trop_larger.json", "t10_trop.json"),
    }
    name_map = {
        "s (Eq.7, deployed)": "s_eq7",
        "GEA ranker": "GEA",
        "conf_VLM only": "conf_VLM",
        "verdict concurrence only": "concurrence",
        "B1 Agent confidence": "B1_single_conf",
        "B2 Mean panel confidence": "B2_mean_conf",
        "B3 Majority vote": "B3_majority_vote",
        "B4 Weighted label agreement": "B4_dawid_skene_transductive",
        "B5 Consolidator (ours)": "B5P_majority_conf_proxy",
    }
    out = {"meta": meta(), "corpora": {}}
    for corpus, (old_p, new_p) in pairs.items():
        old = json.loads(Path(old_p).read_text())
        new = json.loads((OUT / new_p).read_text())
        rows = {}
        for old_name, new_name in name_map.items():
            o = old.get("rows", {}).get(old_name)
            nn = new["ladder"].get(new_name)
            if not o or not nn:
                continue
            rows[new_name] = {
                "old_AURC_allrows_fallback": o["AURC"],
                "new_aurc_fallback_convention": nn["raw"]["aurc_fallback_convention"],
                "new_AURC_attainable": nn["raw"]["AURC_attainable"],
                "new_max_attainable_coverage":
                    nn["raw"]["max_attainable_coverage"],
                "old_SelAcc80": o["SelAcc80"], "new_SelAcc80": nn["raw"]["SelAcc80"],
                "old_Cov99": o["Cov99"], "new_Cov99": nn["raw"]["Cov99"],
                "old_ECE_raw": o["ECE_raw"], "new_ECE10": nn["raw"]["ECE10"],
                "causes": [
                    "eligibility: fallback-scored rows are now forced abstentions, "
                    "never ranked actions",
                    "AURC convention: attainable coverage only",
                    "validity: strict re-parse (invalid JSON / missing confidence "
                    "are permanent abstentions)",
                    "tie rule: deterministic page-id + tie-robust check",
                ],
            }
        old_lbl = "B5 Consolidator (ours)"
        if old_lbl in old.get("rows", {}):
            rows.setdefault("relabels", {})[old_lbl] = \
                "renamed B5-P majority-confidence proxy; the ACTUAL consolidator " \
                "is a new, separately-run mechanism (T9)"
        out["corpora"][corpus] = rows
    (OUT / "old_vs_corrected.json").write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {OUT}/old_vs_corrected.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=list(BUNDLES))
    ap.add_argument("--diffs", action="store_true")
    args = ap.parse_args()
    if args.diffs:
        return run_diffs()
    if not args.corpus:
        print("need --corpus or --diffs")
        return 1
    run_corpus(args.corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
