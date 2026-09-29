"""T4 -- CUE-LEVEL error dependence between panel models (no API calls).

What the reviewers asked for (R2.4, R4.4)
-----------------------------------------
The manuscript reports VERDICT-level error dependence only (errors co-occur on 7.6% of pages
vs 4.0% under independence, ratio ~1.9) and says cue-level dependence is not measured. This
computes the cue-level quantity: for each pair of models and each cue type, how often do the
two models assert the SAME WRONG VALUE, against what an independence model predicts?

"Wrong" is decided by the cue's own verifier, not by a label: a cue is refuted when its tool
scores it below 0.5, verified at or above, and N/A when the tool cannot decide (those are
dropped). The structural and detector tools return exactly 0 or 1, so 0.5 is an arbitrary
midpoint for them; only the perceptual logo tool is continuous, where 0.5 is the threshold the
verifier-soundness evaluation already uses. `--theta` re-runs the whole analysis at another
threshold so the sensitivity is checkable.

Per (pair, cue type) it reports, over the pages where BOTH models assert a value of that type:
    p_i, p_j          marginal rate each model asserts a refuted value
    both_wrong        observed rate both assert a refuted value (any values)
    same_wrong        observed rate both assert the SAME refuted value  <- the agreement-on-error
    indep             p_i * p_j, the independence prediction
    ratio_*           observed / predicted; 1.0 means independent, >1 means correlated errors
and repeats all of it CONDITIONED ON PAGES THE SYSTEM CLASSIFIED INCORRECTLY, which is the
conditional quantity R2 asked for.

Pairs are also grouped by family (Meta / Alibaba / OpenAI) and modality (text / vision), so the
same-family vs cross-family and same-modality vs cross-modal splits fall out. NOTE: the larger
panel is Llama-3.3-70B + Qwen-2.5-72B + GPT-4o, so every pair in it is cross-family; the
same-family contrast only exists on the smaller panel (configs/panel_samefamily.yaml), and
comparing across the two is a cross-panel comparison, not a controlled one.

Usage
    uv run scripts/revision_cue_dependence.py \
        --bundle results/revision/bundle_or_full.jsonl \
        --out results/revision/t4_cue_dependence_or.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.data_io import read_manifest
from phishproof.schema import Cue, CueType
from phishproof.aggregate.normalize import normalize_cue
from phishproof.tools.detector import HtmlBrandDetector
from phishproof.tools.logo_brand import CLIPLogoEmbedder
from phishproof.tools.registry import GroundingContext, ground_cue

# model identity behind each panel slot, for the family / modality splits
PANEL_META = {
    "agent_a_text":  {"family": "Meta",    "modality": "text"},
    "agent_b_text":  {"family": "Alibaba", "modality": "text"},
    "agent_c_vision": {"family": "OpenAI", "modality": "vision"},
}
ACTIVE = ("brand_claim", "form_action_domain", "credential_intent", "logo_brand")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path("results/revision/bundle_or_full.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("data/phishsel_final/test.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/revision/t4_cue_dependence_or.json"))
    ap.add_argument("--theta", type=float, default=0.5, help="refuted if grounding score < theta")
    ap.add_argument("--no-logo", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.bundle.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("agents")]
    pages = {p.page_id: p for p in read_manifest(args.manifest)}
    ctx = GroundingContext(detector=HtmlBrandDetector(),
                           logo_embedder=None if args.no_logo else CLIPLogoEmbedder())
    if not args.no_logo:
        ctx.logo_embedder._ensure()

    # ---- ground every distinct (page, cue) once ------------------------------------
    import threading
    from concurrent.futures import ThreadPoolExecutor

    grounded: dict[str, dict[tuple[str, str], float | None]] = {}
    lock = threading.Lock()
    done = [0]

    def ground_page(r):
        pid = r["page_id"]
        page = pages.get(pid)
        if page is None:
            return pid, {}
        seen: dict[tuple[str, str], float | None] = {}
        for a in r["agents"]:
            for c in a.get("cues", []):
                try:
                    cue = Cue(type=CueType(c["type"]), value=c["value"],
                              raw_value=c["value"], asserted_by=a["id"])
                except ValueError:
                    continue
                nc = normalize_cue(cue)
                if nc is None or nc.key() in seen:
                    continue
                res = ground_cue(nc, page, ctx)
                seen[nc.key()] = res.score if res is not None else None
        return pid, seen

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for pid, seen in ex.map(ground_page, rows):
            grounded[pid] = seen
            with lock:
                done[0] += 1
                if done[0] % 500 == 0:
                    print(f"  grounded {done[0]}/{len(rows)}", flush=True)

    # ---- per page, per agent, per type: the set of asserted normalized values -------
    # value -> refuted? ; N/A values are dropped entirely (the tool cannot decide)
    asserted: dict[str, dict[str, dict[str, set[str]]]] = {}   # pid -> agent -> type -> {values}
    refuted: dict[str, dict[str, dict[str, set[str]]]] = {}    # pid -> agent -> type -> {wrong values}
    for r in rows:
        pid = r["page_id"]
        g = grounded.get(pid, {})
        asserted[pid] = defaultdict(lambda: defaultdict(set))
        refuted[pid] = defaultdict(lambda: defaultdict(set))
        for a in r["agents"]:
            for c in a.get("cues", []):
                try:
                    cue = Cue(type=CueType(c["type"]), value=c["value"],
                              raw_value=c["value"], asserted_by=a["id"])
                except ValueError:
                    continue
                nc = normalize_cue(cue)
                if nc is None:
                    continue
                score = g.get(nc.key())
                if score is None:      # N/A -> not decidable, excluded
                    continue
                asserted[pid][a["id"]][nc.type.value].add(nc.value)
                if score < args.theta:
                    refuted[pid][a["id"]][nc.type.value].add(nc.value)

    correct = {r["page_id"]: (r["verdict"] == r["label"]) for r in rows}
    agents = [a for a in PANEL_META if any(a in asserted[r["page_id"]] for r in rows)]

    def analyse(pids: list[str]) -> dict:
        out: dict = {}
        for i, j in itertools.combinations(agents, 2):
            pair = f"{i}|{j}"
            out[pair] = {
                "family_pair": f"{PANEL_META[i]['family']}+{PANEL_META[j]['family']}",
                "same_family": PANEL_META[i]["family"] == PANEL_META[j]["family"],
                "modality_pair": f"{PANEL_META[i]['modality']}+{PANEL_META[j]['modality']}",
                "same_modality": PANEL_META[i]["modality"] == PANEL_META[j]["modality"],
                "types": {},
            }
            for t in ACTIVE:
                both = [p for p in pids
                        if asserted[p][i].get(t) and asserted[p][j].get(t)]
                n = len(both)
                if n == 0:
                    out[pair]["types"][t] = {"n_both_assert": 0}
                    continue
                wi = sum(1 for p in both if refuted[p][i].get(t))
                wj = sum(1 for p in both if refuted[p][j].get(t))
                bw = sum(1 for p in both if refuted[p][i].get(t) and refuted[p][j].get(t))
                sw = sum(1 for p in both
                         if refuted[p][i].get(t, set()) & refuted[p][j].get(t, set()))
                pi, pj = wi / n, wj / n
                indep = pi * pj
                out[pair]["types"][t] = {
                    "n_both_assert": n,
                    "p_i_wrong": round(pi, 4), "p_j_wrong": round(pj, 4),
                    "indep_pred": round(indep, 4),
                    "both_wrong_obs": round(bw / n, 4), "both_wrong_n": bw,
                    "ratio_both_wrong": round((bw / n) / indep, 2) if indep > 0 else None,
                    "same_wrong_value_obs": round(sw / n, 4), "same_wrong_value_n": sw,
                    "ratio_same_wrong_value": round((sw / n) / indep, 2) if indep > 0 else None,
                }
        return out

    all_pids = [r["page_id"] for r in rows]
    err_pids = [p for p in all_pids if not correct[p]]
    result = {
        "bundle": str(args.bundle),
        "theta": args.theta,
        "n_pages": len(all_pids),
        "n_system_errors": len(err_pids),
        "panel": PANEL_META,
        "note": ("Larger panel is Meta+Alibaba+OpenAI, so every pair is cross-family; "
                 "the same-family contrast requires configs/panel_samefamily.yaml "
                 "(smaller panel) and is a cross-panel comparison."),
        "all_pages": analyse(all_pids),
        "conditioned_on_system_error": analyse(err_pids),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
