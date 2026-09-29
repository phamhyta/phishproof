"""T3 -- gate attribution under attack on the larger panel, as a strict partition.

Each EVADED page (label=phish, clean verdict=phish, attacked verdict=benign) lands in
exactly one bucket, so the three columns sum to the evaded count by construction:

  score_abstain : calibrated trust kappa(s) <  tau         -> the score gate rejected it
  veto_only     : calibrated trust kappa(s) >= tau AND the verification gate failed
                  (consensus groundedness G < g_min)       -> only the veto stopped it
  wrong_act     : passed BOTH gates and was acted on with a flipped verdict

This replaces `postprocess_adversarial_or.py`'s `abstain_veto_only_pct`, which tested only
`G < g_min` and never checked that the score gate had accepted the page first, so it counted
score-gate rejections as veto saves and could report 100% for both columns at once.

Also reports the abstention rate on the MATCHED CLEAN pages at the same threshold -- the base
rate the attacked numbers must be read against -- and Wilson 95% intervals on every rate, so
the paper never again reports a bare "95-100% abstain".

`trust` in the bundle is already the calibrated value: the bundle's clean formula 0.9005 maps
to trust 0.9444 under `results/calibrator_or.json`, so tau is applied to kappa(s), not raw s.

Usage
    uv run scripts/revision_adversarial_or.py --out results/revision/t3_adversarial_or.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TAU_DEFAULT = 0.9651  # the value in code; the manuscript's 0.965 is a rounding of it
G_MIN_DEFAULT = 1.0   # strict grounding veto (Prop. 2)


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def rate(k: int, n: int) -> dict:
    return {"n": k, "pct": round(100 * k / n, 1) if n else None, "ci95": wilson(k, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path,
                    default=Path("results/_or/bundle_adversarial_or.jsonl"))
    ap.add_argument("--clean-g", type=Path, default=Path("results/_or/clean_g.json"))
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument("--g-min", type=float, default=G_MIN_DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("results/revision/t3_adversarial_or.json"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]
    clean_g = json.loads(args.clean_g.read_text()) if args.clean_g.exists() else {}

    def g_of(r: dict, side: str) -> float:
        if side == "clean" and r["page_id"] in clean_g:
            return float(clean_g[r["page_id"]])
        v = r[side].get("G")
        return float(v) if v is not None else 1.0

    by_attack: dict[str, list[dict]] = {}
    for r in rows:
        by_attack.setdefault(r["attack"], []).append(r)

    summary: dict = {"tau": args.tau, "g_min": args.g_min,
                     "definition": {
                         "score_abstain": "trust < tau",
                         "veto_only": "trust >= tau and G < g_min",
                         "wrong_act": "trust >= tau and G >= g_min"},
                     "attacks": {}}

    for attack, rs in sorted(by_attack.items()):
        n = len(rs)
        # evaded = was caught clean, is missed under attack
        evaded = [r for r in rs
                  if r["label"] == "phish"
                  and r["clean"]["verdict"] == "phish"
                  and r["attacked"]["verdict"] != "phish"]
        ne = len(evaded)

        score_abstain = [r for r in evaded if r["attacked"]["trust"] < args.tau]
        passed_score = [r for r in evaded if r["attacked"]["trust"] >= args.tau]
        veto_only = [r for r in passed_score if g_of(r, "attacked") < args.g_min]
        wrong_act = [r for r in passed_score if g_of(r, "attacked") >= args.g_min]

        assert len(score_abstain) + len(veto_only) + len(wrong_act) == ne, attack

        # base rate: abstention on the MATCHED CLEAN pages at the same threshold
        clean_abstain = [r for r in rs if r["clean"]["trust"] < args.tau]

        summary["attacks"][attack] = {
            "n": n,
            "evaded": rate(ne, n),
            "score_abstain": rate(len(score_abstain), ne),
            "veto_only": rate(len(veto_only), ne),
            "wrong_act": rate(len(wrong_act), ne),
            # the paper's "Abstain" column = stopped by EITHER gate = score_abstain + veto_only
            "abstain_either_gate": rate(len(score_abstain) + len(veto_only), ne),
            "partition_sums_to_evaded": len(score_abstain) + len(veto_only) + len(wrong_act) == ne,
            "clean_abstain_base_rate": rate(len(clean_abstain), n),
            "G_clean_mean": round(float(np.mean([g_of(r, "clean") for r in rs])), 3),
            "G_attacked_mean": round(float(np.mean([g_of(r, "attacked") for r in rs])), 3),
            "G_attacked_mean_evaded": (round(float(np.mean([g_of(r, "attacked") for r in evaded])), 3)
                                       if ne else None),
            "trust_evaded_mean": (round(float(np.mean([r["attacked"]["trust"] for r in evaded])), 3)
                                  if ne else None),
        }

    # pooled over attacks
    all_ev = [r for r in rows if r["label"] == "phish" and r["clean"]["verdict"] == "phish"
              and r["attacked"]["verdict"] != "phish"]
    sa = [r for r in all_ev if r["attacked"]["trust"] < args.tau]
    ps = [r for r in all_ev if r["attacked"]["trust"] >= args.tau]
    vo = [r for r in ps if g_of(r, "attacked") < args.g_min]
    wa = [r for r in ps if g_of(r, "attacked") >= args.g_min]
    summary["pooled"] = {
        "n": len(rows), "evaded": rate(len(all_ev), len(rows)),
        "score_abstain": rate(len(sa), len(all_ev)),
        "veto_only": rate(len(vo), len(all_ev)),
        "wrong_act": rate(len(wa), len(all_ev)),
        "abstain_either_gate": rate(len(sa) + len(vo), len(all_ev)),
        "clean_abstain_base_rate": rate(sum(1 for r in rows if r["clean"]["trust"] < args.tau),
                                        len(rows)),
    }

    text = json.dumps(summary, indent=2)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
