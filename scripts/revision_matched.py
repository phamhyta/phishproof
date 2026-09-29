#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Matched-panel reliability comparison for the COMPELECENG revision.

Every reliability score is computed from ONE bundle of panel outputs, on ONE page set,
under ONE calibration protocol, so a difference between rows is a difference between
trust mechanisms and nothing else. This is the controlled contrast reviewers R1-R5 all
asked for.

Two rules that are easy to get wrong and that the reviewers checked:

  1. Pages whose required model output is missing are FORCED ABSTENTIONS. They stay in
     the coverage denominator and rank last. They are never deleted from the metric.
  2. Calibration is CROSS-FITTED and identical for every method: the split is divided
     into K folds and each fold's isotonic map is fitted on the other K-1. No method is
     calibrated on its own test rows, and no method gets a calibrator another does not.

Usage
    uv run scripts/revision_matched.py artifacts/bundle_final.jsonl           # matched panel
    uv run scripts/revision_matched.py results/bundle_or.jsonl --out out.json # larger panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

VISION_ID = "agent_c_vision"
EPS = 1e-3  # tie-break weight in Eq. 7
N_BOOT = 2000
N_FOLDS = 5

# paper-facing baseline label -> key inside the bundle's "baselines" dict
BASELINE_MAP = {
    "B1 Agent confidence": "B1",
    "B2 Mean panel confidence": "B2",
    "B3 Majority vote": "B4",
    "B4 Weighted label agreement": "B5",
    "B5 Consolidator (ours)": "B6",
}


# ----------------------------------------------------------------- bundle accessors
def agent(row: dict, aid: str) -> dict:
    for a in row.get("agents", []):
        if a.get("id") == aid:
            return a
    return {"verdict": None, "confidence": None}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


# ----------------------------------------------------------------- metrics
def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve, x100. NaN ranks last (forced abstention).

    Ties are resolved by input order (stable sort). That is the published definition, kept
    here so the larger-panel rows stay comparable with the matched-panel rows already pinned
    in matched-reanalysis.json. It is NOT label-independent: the panel emits only a handful
    of distinct confidence levels, so tied blocks are large and their internal order is an
    artefact of manifest order. Use aurc_tie_robust() to check that a conclusion does not
    depend on it.
    """
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, kind="mergesort")
    err = np.cumsum(1.0 - correct[order])
    return 100.0 * float(np.mean(err / np.arange(1, len(order) + 1)))


def aurc_tie_robust(score: np.ndarray, correct: np.ndarray, n_perm: int = 200,
                    seed: int = 0) -> tuple[float, float]:
    """Expected AURC under a RANDOM ordering within each tied score block.

    Label-independent by construction: the tie order cannot encode the label, so a score
    with many ties can no longer be flattered (or punished) by the manifest happening to
    put its correct pages first. Returns (mean, std) over n_perm random tie orderings.
    """
    s = np.where(np.isnan(score), -np.inf, score)
    rng = np.random.RandomState(seed)
    n = len(s)
    idx = np.arange(1, n + 1)
    vals = []
    for _ in range(n_perm):
        jitter = rng.permutation(n)                     # random, label-independent
        order = np.lexsort((jitter, -s))                # primary -s, ties broken at random
        err = np.cumsum(1.0 - correct[order])
        vals.append(100.0 * float(np.mean(err / idx)))
    return float(np.mean(vals)), float(np.std(vals))


def sel_acc(score: np.ndarray, correct: np.ndarray, coverage: float = 0.80) -> float:
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, kind="mergesort")
    k = max(1, int(round(coverage * len(order))))
    return 100.0 * float(np.mean(correct[order[:k]]))


def cov_at(score: np.ndarray, correct: np.ndarray, target: float = 0.99) -> float:
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, kind="mergesort")
    running = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(running >= target)[0]
    return float(ok.max() + 1) / len(order) if len(ok) else 0.0


def ece(prob: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    p = np.clip(np.where(np.isnan(prob), 0.0, prob), 0.0, 1.0)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        hi_incl = i == bins - 1
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if hi_incl else (p < edges[i + 1]))
        if m.sum():
            total += m.sum() / len(p) * abs(correct[m].mean() - p[m].mean())
    return float(total)


GROUPS: np.ndarray | None = None  # set from --groups; duplicate-group id per row


def _make_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Random folds, or GROUP-DISJOINT folds when --groups is supplied.

    With random folds a near-duplicate of a test page can sit in the folds its calibration
    map is fitted on, so the map is partly fitted on the page it then scores. 42.8% of the
    Phishpedia test split shares a duplicate group with another test row (T5), so this is
    not a corner case. Group-disjoint folds put a whole duplicate group on one side.
    """
    rng = np.random.RandomState(seed)
    if GROUPS is None:
        return np.array_split(rng.permutation(n), k)
    uniq = np.array(sorted(set(GROUPS.tolist())))
    assign = {g: i % k for i, g in enumerate(rng.permutation(uniq))}
    which = np.array([assign[g] for g in GROUPS])
    return [np.where(which == i)[0] for i in range(k)]


def crossfit_isotonic(score: np.ndarray, correct: np.ndarray, k: int = N_FOLDS,
                      seed: int = 0) -> np.ndarray:
    """Identical calibration protocol for every method; no method sees its own test rows."""
    s = np.where(np.isnan(score), np.nanmin(score) - 1.0, score)
    folds = _make_folds(len(s), k, seed)
    out = np.zeros(len(s))
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(s[train], correct[train])
        out[test] = ir.predict(s[test])
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--groups", type=Path, default=None,
                    help="t5_grouped_split.json; use GROUP-DISJOINT calibration folds")
    ap.add_argument("--dedup", action="store_true",
                    help="keep one row per duplicate group (needs --groups)")
    ap.add_argument("--tau", type=float, default=0.9651,
                    help="operating point in CALIBRATED space; 0.9651 is the code value that\n                         the manuscript rounds to 0.965")
    args = ap.parse_args()

    rows = load(args.bundle)

    global GROUPS
    if args.groups:
        gmap = json.loads(args.groups.read_text())["test_group_map"]
        rows = [r for r in rows if r["page_id"] in gmap]
        if args.dedup:  # one representative row per duplicate group, first occurrence
            seen: set[str] = set()
            kept = []
            for r in rows:
                g = gmap[r["page_id"]]
                if g not in seen:
                    seen.add(g)
                    kept.append(r)
            rows = kept
        GROUPS = np.array([gmap[r["page_id"]] for r in rows])
        print(f"[groups] {len(rows)} rows, {len(set(GROUPS.tolist()))} duplicate-groups, "
              f"dedup={args.dedup}")

    n = len(rows)
    rng = np.random.RandomState(0)

    correct = np.array([r["verdict"] == r["label"] for r in rows], dtype=float)
    is_phish = np.array([r["label"] == "phish" for r in rows], dtype=float)
    pred_phish = np.array([r["verdict"] == "phish" for r in rows], dtype=float)

    # conf_VLM. A page with no vision confidence gets the deployed fallback 0.5, which is
    # below every confidence the panel ever emits (0.9 / 0.95 / 1.0), so it still ranks last
    # and stays in the coverage denominator as a forced abstention.
    conf_raw = np.array([
        agent(r, VISION_ID)["confidence"]
        if agent(r, VISION_ID)["confidence"] is not None else np.nan
        for r in rows
    ])
    conf = np.where(np.isnan(conf_raw), 0.5, conf_raw)

    # BUGFIX: a_text is "the fraction of text models agreeing with THE VISION MODEL'S verdict"
    # (method.tex eq. 7, and scripts/run_adversarial_or.py::formula_trust, which is what the
    # deployed detector and the adversarial pipeline actually run). This previously compared
    # each text model against the PANEL's majority verdict, a different quantity. On the 3B
    # panel the two coincide -- its text agents are near-constant, so the majority is the
    # vision verdict -- which is why the matched-panel number barely moves (1.27 -> 1.26). On
    # the larger panel the text models genuinely dissent and the two diverge sharply
    # (AURC 2.63 under the old reading vs 1.80 under eq. 7).
    def _a_text(r: dict) -> float:
        ref = agent(r, VISION_ID)["verdict"] or r["verdict"]
        text = [a for a in r["agents"] if a["id"] != VISION_ID]
        return sum(1 for a in text if a["verdict"] == ref) / len(text) if text else 0.0

    a_text = np.array([_a_text(r) for r in rows])
    failed = np.array([bool(r.get("replay_failed")) for r in rows])
    gea = np.array([np.nan if r.get("replay_failed") else r["gea"] for r in rows])
    ground = np.array([np.nan if r.get("replay_failed") else r.get("groundedness", np.nan)
                       for r in rows])
    frac = np.array([np.nan if r.get("replay_failed") else r.get("agreement", np.nan)
                     for r in rows])
    n_cue = np.array([len(r.get("consensus_cues", [])) for r in rows])
    s_eq7 = conf + EPS * a_text

    out: dict = {
        "bundle": str(args.bundle),
        "n": n,
        "accuracy": round(100 * float(correct.mean()), 2),
        "forced_abstentions": int(np.isnan(conf_raw).sum()),
        "unrecoverable_rows": int(failed.sum()),
    }

    def summarise(score: np.ndarray) -> dict:
        p = crossfit_isotonic(score, correct)
        raw = np.clip(np.where(np.isnan(score), 0.0, score), 0.0, 1.0)
        tr_mean, tr_std = aurc_tie_robust(score, correct)
        # Plan T1 rule 4: report the selective metrics BOTH BEFORE AND AFTER the common
        # calibration. Ranking by the raw score is not neutral -- isotonic regression fitted
        # on held-out folds can CORRECT a non-monotone region of a raw score (it merges or
        # reorders levels whose empirical accuracy does not follow the score). A method whose
        # raw ordering is slightly non-monotone is therefore penalised by the raw-AURC column
        # and recovers under the calibrated one. Since every method gets the identical
        # cross-fitted protocol, the calibrated column is the like-for-like comparison.
        tr_mean_c, tr_std_c = aurc_tie_robust(p, correct)
        return {
            "AURC": round(aurc(score, correct), 2),
            "AURC_tie_robust": round(tr_mean, 2),
            "AURC_tie_robust_sd": round(tr_std, 3),
            "AURC_calibrated": round(aurc(p, correct), 2),
            "AURC_calibrated_tie_robust": round(tr_mean_c, 2),
            "AURC_calibrated_tie_robust_sd": round(tr_std_c, 3),
            "SelAcc80": round(sel_acc(score, correct), 1),
            "SelAcc80_calibrated": round(sel_acc(p, correct), 1),
            "Cov99": round(cov_at(score, correct), 3),
            "Cov99_calibrated": round(cov_at(p, correct), 3),
            "ECE_raw": round(ece(raw, correct), 3),
            "ECE_matched": round(ece(p, correct), 3),
        }

    # ---- main table + attribution ladder, all on the same rows -------------
    scores: dict[str, np.ndarray] = {
        # A page whose panel output could not be recovered has no baseline value either.
        # It becomes NaN, which ranks last -- a forced abstention that stays in the
        # coverage denominator, exactly as for the trust scores (plan T1 rule 3).
        label: np.array([r.get("baselines", {}).get(key, np.nan) for r in rows], dtype=float)
        for label, key in BASELINE_MAP.items()
        if any(key in r.get("baselines", {}) for r in rows)
    }
    scores.update({
        "GEA ranker": gea,
        "s (Eq.7, deployed)": s_eq7,
        "conf_VLM only": conf,
        "verdict concurrence only": a_text,
        "consensus fraction": frac,
        "groundedness only": ground,
        "GEA x groundedness": gea * ground,
    })
    out["rows"] = {k: summarise(v) for k, v in scores.items() if not np.all(np.isnan(v))}

    # ---- paired page-level bootstrap against the deployed score ------------
    boot = [rng.randint(0, n, n) for _ in range(N_BOOT)]

    def paired(a: np.ndarray, b: np.ndarray) -> dict:
        d = np.array([aurc(a[i], correct[i]) - aurc(b[i], correct[i]) for i in boot])
        return {
            "delta": round(float(d.mean()), 2),
            "ci": [round(float(np.percentile(d, 2.5)), 2),
                   round(float(np.percentile(d, 97.5)), 2)],
            "p_one_sided": round(float((d <= 0).mean()), 4),
        }

    cal_scores = {k: crossfit_isotonic(v, correct) for k, v in scores.items()
                  if not np.all(np.isnan(v))}
    out["paired_vs_s_calibrated"] = {
        k: paired(v, cal_scores["s (Eq.7, deployed)"]) for k, v in cal_scores.items()
        if k != "s (Eq.7, deployed)"
    }
    out["paired_vs_s"] = {
        k: paired(v, s_eq7) for k, v in scores.items()
        if k != "s (Eq.7, deployed)" and not np.all(np.isnan(v))
    }
    out["paired_vs_gea"] = {
        k: paired(v, gea) for k in list(BASELINE_MAP) if k in scores
        for v in [scores[k]]
    }

    # ---- complete-policy operating point + gate attribution ----------------
    rho = crossfit_isotonic(s_eq7, correct)
    score_gate = rho >= args.tau
    verified = (n_cue > 0) & (ground >= 0.999)
    act = score_gate & verified
    out["gate"] = {
        "tau": args.tau,
        "score_gate_acts": int(score_gate.sum()),
        "score_gate_risk_pct": round(100 * float(1 - correct[score_gate].mean()), 2)
        if score_gate.sum() else None,
        "veto_removes": int((score_gate & ~verified).sum()),
        "final_acts": int(act.sum()),
        "final_risk_pct": round(100 * float(1 - correct[act].mean()), 2) if act.sum() else None,
        "action_coverage_pct": round(100 * float(act.mean()), 1),
    }

    # ---- prevalence reweighting -------------------------------------------
    fp = float(((pred_phish == 1) & (is_phish == 0) & act).sum())
    n_benign = float((1 - is_phish).sum())
    tpr = float(((pred_phish == 1) & (is_phish == 1) & act).sum()) / max(float(is_phish.sum()), 1)
    _, fpr_hi = wilson(int(fp), int(n_benign))
    cov_p, cov_b = float(act[is_phish == 1].mean()), float(act[is_phish == 0].mean())
    out["base_rate"] = {
        "block_rate_phish_pct": round(100 * tpr, 2),
        "false_blocks": int(fp),
        "n_benign": int(n_benign),
        "fpr_wilson_hi_pct": round(100 * fpr_hi, 4),
        # PPV is bounded, not estimated, when the observed false-block count is zero
        "ppv_lower_bound_pct": {
            str(pv): round(100 * (pv * tpr) / max(pv * tpr + (1 - pv) * fpr_hi, 1e-12), 1)
            for pv in (0.1, 0.01, 0.001)
        },
        "escalation_pct": {
            str(pv): round(100 * (1 - (pv * cov_p + (1 - pv) * cov_b)), 1)
            for pv in (0.1, 0.01, 0.001)
        },
    }

    # ---- panel composition -------------------------------------------------
    lab = (is_phish == 1).astype(int)
    verdicts = {
        a["id"]: np.array([1 if agent(r, a["id"])["verdict"] == "phish" else 0 for r in rows])
        for a in rows[0]["agents"]
    }

    def kappa(x: np.ndarray, y: np.ndarray) -> float | None:
        po = (x == y).mean()
        pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
        return round(float((po - pe) / (1 - pe)), 3) if pe < 1 else None

    ids = list(verdicts)
    out["panel"] = {
        "accuracy_pct": {i: round(100 * float((verdicts[i] == lab).mean()), 2) for i in ids},
        "verdict_kappa": {f"{ids[i]}|{ids[j]}": kappa(verdicts[ids[i]], verdicts[ids[j]])
                          for i in range(len(ids)) for j in range(i + 1, len(ids))},
        "pages_with_consensus_cue_pct": round(100 * float((n_cue > 0).mean()), 1),
        "conf_levels": int(len(set(conf[~np.isnan(conf)].tolist()))),
        "eq7_levels": int(len(set(np.round(s_eq7[~np.isnan(s_eq7)], 9).tolist()))),
    }

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
