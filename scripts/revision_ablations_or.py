"""T2 -- component ablations on the larger panel, re-scored from the replayed bundle.

Every row here is currently reported on the matched (3B) panel only, and exp_rq3.tex carries a
hedge saying the magnitudes may not transfer. This measures them on the larger panel from
results/revision/bundle_or_full.jsonl, which carries the per-agent CUE SETS, so each variant is
a re-score of the same cached outputs -- no API calls.

Variants
  full                : per-type GEA over the whole panel                       (headline GEA)
  -vision             : per-type GEA over the TEXT agents only
  -per-type (global)  : global consensus fraction |cons|/|E| instead of per-type agreement
  M=1 / M=2 / M=3     : panel-size sweep, per-type GEA over agent subsets
  k sweep             : consensus threshold k in {1,2,3} for the consensus fraction
  learned combiner    : logistic regression on [conf_VLM, a_text, GEA, G, consensus fraction],
                        CROSS-FITTED (5 folds) so no page is scored by a model fitted on it,
                        against the uniform (parameter-free) combination. The fitted
                        coefficients are written out, which is the refit that was never pinned.

AURC is reported under the published stable-sort tie rule and under a label-independent random
tie rule, because several of these scores are coarse and heavily tied.

Usage
    uv run scripts/revision_ablations_or.py --out results/revision/t2_ablations_or.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishproof.aggregate.normalize import normalize_value  # noqa: E402
from phishproof.schema import CueType  # noqa: E402

VIS = "agent_c_vision"
ACTIVE = ("brand_claim", "form_action_domain", "credential_intent", "logo_brand")


def aurc(score, correct):
    s = np.where(np.isnan(np.asarray(score, float)), -np.inf, np.asarray(score, float))
    o = np.argsort(-s, kind="stable")
    c = np.asarray(correct, float)[o]
    return 100 * float(np.mean(1 - np.cumsum(c) / np.arange(1, len(c) + 1)))


def aurc_tr(score, correct, n_perm=200, seed=0):
    s = np.where(np.isnan(np.asarray(score, float)), -np.inf, np.asarray(score, float))
    c = np.asarray(correct, float)
    rng = np.random.RandomState(seed)
    n = len(s)
    k = np.arange(1, n + 1)
    v = [100 * float(np.mean(1 - np.cumsum(c[np.lexsort((rng.permutation(n), -s))]) / k))
         for _ in range(n_perm)]
    return float(np.mean(v)), float(np.std(v))


def sel_acc(score, correct, cov=0.80):
    s = np.where(np.isnan(np.asarray(score, float)), -np.inf, np.asarray(score, float))
    o = np.argsort(-s, kind="stable")
    k = max(1, int(round(cov * len(o))))
    return 100 * float(np.mean(np.asarray(correct, float)[o[:k]]))


def per_type_agreement(cues_by_agent: dict[str, set[tuple[str, str]]]) -> float:
    """GEA = mean over the active cue types of (max agents citing one value) / M."""
    m = len(cues_by_agent)
    if m == 0:
        return 0.0
    total = 0.0
    for t in ACTIVE:
        by_val: dict[str, set[str]] = {}
        for aid, cues in cues_by_agent.items():
            for ct, cv in cues:
                if ct == t:
                    by_val.setdefault(cv, set()).add(aid)
        total += (max((len(s) for s in by_val.values()), default=0) / m)
    return total / len(ACTIVE)


def consensus_fraction(cues_by_agent: dict[str, set[tuple[str, str]]], k: int) -> float:
    """|cues asserted by >= k agents| / |union of cues|."""
    pool: dict[tuple[str, str], set[str]] = {}
    for aid, cues in cues_by_agent.items():
        for c in cues:
            pool.setdefault(c, set()).add(aid)
    if not pool:
        return 0.0
    return sum(1 for v in pool.values() if len(v) >= k) / len(pool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path("results/revision/bundle_or_full.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/revision/t2_ablations_or.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("verdict") is not None]
    correct = np.array([r["verdict"] == r["label"] for r in rows], dtype=float)

    # Cue values must be NORMALIZED before consensus, exactly as score_page does via
    # normalize_cue (eTLD+1 for domains, canonical lexicon for brands, yes/no for
    # credential intent). Comparing raw surface strings silently understates agreement.
    def _norm(c: dict) -> tuple[str, str] | None:
        try:
            t = CueType(c["type"])
        except ValueError:
            return None
        v = normalize_value(t, c["value"])
        return (t.value, v) if v else None

    cues = [{a["id"]: {n for c in a.get("cues", []) if (n := _norm(c))}
             for a in r["agents"]} for r in rows]
    ids = sorted(cues[0]) if cues else []
    text_ids = [i for i in ids if i != VIS]

    conf = np.array([next((a["confidence"] for a in r["agents"] if a["id"] == VIS), None)
                     or 0.5 for r in rows])
    a_text = np.array([
        (lambda ref, t: sum(1 for x in t if x["verdict"] == ref) / len(t) if t else 0.0)(
            next((a["verdict"] for a in r["agents"] if a["id"] == VIS), r["verdict"]),
            [a for a in r["agents"] if a["id"] != VIS])
        for r in rows])
    gea = np.array([r["gea"] for r in rows])
    ground = np.array([r["groundedness"] for r in rows])
    frac = np.array([r["agreement"] for r in rows])

    def row(score):
        # Round before ranking: these scores are coarse rationals (k/M, means of them) and a
        # 1e-16 float difference between two mathematically equal scores otherwise breaks a
        # tie differently and moves AURC by ~0.02 under the stable-sort tie rule.
        score = np.round(np.asarray(score, float), 12)
        m, sd = aurc_tr(score, correct)
        return {"AURC": round(aurc(score, correct), 2),
                "AURC_tie_robust": round(m, 2), "AURC_tie_robust_sd": round(sd, 3),
                "SelAcc80": round(sel_acc(score, correct), 1)}

    out: dict = {"bundle": str(args.bundle), "n": len(rows),
                 "accuracy": round(100 * float(correct.mean()), 2), "variants": {}}
    V = out["variants"]

    V["full (per-type GEA, M=3)"] = row(gea)
    V["-vision (per-type GEA, text agents only)"] = row(
        np.array([per_type_agreement({i: c[i] for i in text_ids}) for c in cues]))
    V["-per-type (global consensus fraction)"] = row(frac)

    # Panel-size sweep. Each subset is scored AS ITS OWN PANEL of size M and evaluated
    # separately; the summary is the mean of the per-subset AURCs. Averaging the SCORES
    # across subsets instead would build a new ensemble over all M agents, which is not a
    # panel of size M and understates the cost of shrinking the panel.
    out["panel_size_sweep"] = {}
    for msize in (1, 2, 3):
        subs = list(itertools.combinations(ids, msize))
        per_sub = {}
        for sub in subs:
            sc = np.array([per_type_agreement({i: c[i] for i in sub}) for c in cues])
            per_sub["+".join(x.replace("agent_", "") for x in sub)] = row(sc)
        out["panel_size_sweep"][f"M={msize}"] = per_sub
        V[f"M={msize} (mean of {len(subs)} per-subset AURCs)"] = {
            "AURC": round(float(np.mean([v["AURC"] for v in per_sub.values()])), 2),
            "AURC_tie_robust": round(float(np.mean([v["AURC_tie_robust"] for v in per_sub.values()])), 2),
            "AURC_tie_robust_sd": None,
            "SelAcc80": round(float(np.mean([v["SelAcc80"] for v in per_sub.values()])), 1)}

    # consensus threshold k sweep
    for k in (1, 2, 3):
        V[f"k={k} (consensus fraction)"] = row(
            np.array([consensus_fraction(c, k) for c in cues]))

    # Learned vs uniform combiner. tab_hfgea's contrast is over the PER-TYPE AGREEMENT
    # SIGNALS a_t only -- "All combiners use the identical per-type agreement features" --
    # so the features are the four a_t, and the uniform average of them IS the GEA score.
    # Using a richer feature set would be a different experiment, not this ablation.
    def per_type_vec(cba: dict[str, set[tuple[str, str]]]) -> list[float]:
        m = len(cba) or 1
        v = []
        for t in ACTIVE:
            by_val: dict[str, set[str]] = {}
            for aid, cs in cba.items():
                for ct, cv in cs:
                    if ct == t:
                        by_val.setdefault(cv, set()).add(aid)
            v.append(max((len(x) for x in by_val.values()), default=0) / m)
        return v

    X = np.array([per_type_vec(c) for c in cues])
    uniform = X.mean(axis=1)
    folds = np.array_split(np.random.RandomState(0).permutation(len(X)), 5)
    learned = np.zeros(len(X))
    coefs = []
    for i in range(5):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(5) if j != i])
        lr = LogisticRegression(max_iter=2000).fit(X[tr], correct[tr])
        learned[te] = lr.predict_proba(X[te])[:, 1]
        coefs.append({"coef": lr.coef_[0].round(4).tolist(),
                      "intercept": round(float(lr.intercept_[0]), 4)})
    V["uniform average of a_t (= GEA, ours)"] = row(uniform)
    V["learned weights on a_t (logistic, cross-fitted)"] = row(learned)
    out["learned_combiner"] = {
        "features": [f"a_t[{t}]" for t in ACTIVE],
        "protocol": "5-fold cross-fitted logistic regression; no page scored by a fold fitted on it",
        "folds": coefs,
        "mean_coef": np.mean([c["coef"] for c in coefs], axis=0).round(4).tolist(),
    }

    text = json.dumps(out, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"{'variant':46} {'AURC':>6} {'tie-rob':>8} {'SelAcc80':>9}")
    print("-" * 73)
    for k, v in V.items():
        print(f"{k:46} {v['AURC']:>6.2f} {v['AURC_tie_robust']:>8.2f} {v['SelAcc80']:>9.1f}")
    print("\nper-subset panel-size sweep:")
    for m, d in out["panel_size_sweep"].items():
        for sub, v in d.items():
            print(f"   {m:5} {sub:38} AURC={v['AURC']:>6.2f}  SelAcc80={v['SelAcc80']:>5.1f}")
    print(f"\nlearned combiner mean coef: {out['learned_combiner']['mean_coef']}")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
