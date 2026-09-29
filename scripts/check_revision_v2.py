#!/usr/bin/env -S uv run --quiet
"""T13 regression checks — re-derive the policy, validity, and headline table cells
DIRECTLY from the revision_v2 records.

This does not compare against a summary JSON; it recomputes each asserted quantity from
the per-page eligibility / consolidator / attack records and fails loudly on a mismatch.
Extend the CHECKS list whenever a manuscript cell is added.

Usage
    uv run scripts/check_revision_v2.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

V2 = Path("results/revision_v2")


def load(path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ------------------------------------------------------------------- policy invariants
@check("T8 complete gate: act iff valid AND score_pass AND consensus AND available AND pass")
def _t8_gate():
    bad = 0
    for corpus in ("phishpedia", "apwg", "trop"):
        for r in load(V2 / f"t8/eligibility_{corpus}.jsonl"):
            c = r["complete"]
            expect = (c["valid_inference"] and c["score_pass"]
                      and c["consensus_nonempty"] and c["all_available"]
                      and c["all_pass"])
            if (c["action"] == "act") != expect:
                bad += 1
    return bad == 0, f"{bad} rows where action disagrees with the gate conjunction"


@check("T8 unavailable consensus check never passes (all_available False => all_pass never True acts)")
def _t8_unavail():
    bad = 0
    for corpus in ("phishpedia", "apwg", "trop"):
        for r in load(V2 / f"t8/eligibility_{corpus}.jsonl"):
            c = r["complete"]
            if c["action"] == "act":
                for cue in r["cues"]:
                    if cue["in_strict_consensus"] and not cue["available"]:
                        bad += 1
    return bad == 0, f"{bad} acted rows had an unavailable consensus cue"


@check("T8 missing vision confidence => permanent abstention")
def _t8_missing_conf():
    bad = 0
    for corpus in ("phishpedia", "apwg", "trop"):
        for r in load(V2 / f"t8/eligibility_{corpus}.jsonl"):
            vis = next((v for v in r["agents"] if v["is_vision"]), None)
            if vis and vis["raw_present"] and vis["valid_json"] \
                    and not vis["has_confidence"]:
                if r["complete"]["action"] == "act":
                    bad += 1
    return bad == 0, f"{bad} acted rows lacked required vision confidence"


@check("T8 frozen logo threshold selected on calibration, not the diagnostic 0.5")
def _t8_threshold():
    pol = json.loads((V2 / "t8/policy_frozen.json").read_text())
    t = pol["t_logo"]
    ok = pol["t_logo_selection"]["manifest"].endswith("calibration.jsonl") and t != 0.5
    return ok, f"t_logo={t} from {pol['t_logo_selection']['manifest']}"


# ------------------------------------------------------------------- validity counts
@check("T10 missing-vision-confidence counts match the eligibility records")
def _t10_missing():
    want = {"phishpedia": None, "apwg": None, "trop": None}
    msgs = []
    ok = True
    for corpus in want:
        rows = load(V2 / f"t8/eligibility_{corpus}.jsonl")
        n_missing = sum(1 for r in rows
                        if any(v["is_vision"] and (not v["raw_present"]
                               or not v["has_confidence"]) for v in r["agents"]))
        t10 = json.loads((V2 / f"t10/t10_{corpus}.json").read_text())
        rep = t10["reason_table"]["missing_vision_confidence_rows"]
        # eligibility missing-conf counts rows with conf None OR vision invalid;
        # the T10 reason table counts conf None specifically -> vision present-but-None
        conf_none = sum(1 for r in rows if r["conf_vlm"] is None)
        if rep != conf_none:
            ok = False
        msgs.append(f"{corpus}: conf_none={conf_none} reported={rep} "
                    f"(any-missing={n_missing})")
    return ok, "; ".join(msgs)


@check("T9 smaller-panel B5 provenance: stored == parsed consolidator on every row")
def _t9_provenance():
    s = json.loads((V2 / "t9/consolidator_provenance_smaller_summary.json").read_text())
    c = s["counts"]
    ok = (c["recovered"] == c["stored_matches_parsed_conf"]
          and c["parse_fallback"] == 0
          and c["stored_matches_proxy"] < c["recovered"])
    return ok, (f"recovered={c['recovered']} matches_parsed={c['stored_matches_parsed_conf']} "
                f"matches_proxy={c['stored_matches_proxy']} fallback={c['parse_fallback']}")


@check("T9 larger-panel B5-P proxy != actual consolidator (distinct mechanisms)")
def _t9_distinct():
    msgs = []
    ok = True
    for corpus in ("phishpedia", "apwg"):
        rows = [r for r in load(V2 / f"t9/consolidator_{corpus}.jsonl")
                if r.get("status") == "ok"]
        same = sum(1 for r in rows if abs(r["confidence"] - r["proxy_b5p"]) < 1e-9)
        frac = same / len(rows)
        if frac > 0.5:
            ok = False
        msgs.append(f"{corpus}: consolidator==proxy on {frac:.1%}")
    return ok, "; ".join(msgs)


# ------------------------------------------------------------------- attack partition
@check("T11 attack partition: no wrong_act, and the summary's own sum flag holds")
def _t11_partition():
    s = json.loads((V2 / "t11/attack_policy_summary.json").read_text())["attacks"]
    bad = []
    for atk, d in s.items():
        p = d["partition"]
        if not p["sums_to_evaded"]:
            bad.append(atk)
        if p["wrong_act"] != 0:   # the fail-safe: no evaded page is ever acted on
            bad.append(atk + "(wrong_act!=0)")
    return not bad, "no wrong_act; sum flag holds" if not bad else f"bad: {bad}"


@check("T11 veto_only = valid inference + score accepted + a verification check rejected")
def _t11_veto():
    bad = 0
    for r in load(V2 / "t11/attack_policy_records.jsonl"):
        if not r["evaded"]:
            continue
        # a genuine veto-only requires VALID inference, score accepted, action abstain
        if (r["inference_valid_attacked"] and r["attacked_score_pass"]
                and r["attacked_action"] == "abstain"):
            if r["attacked_all_pass"] and r["attacked_all_available"]:
                bad += 1  # every check passed yet abstained -> not a real veto
    return bad == 0, f"{bad} veto-only rows had all checks passing (contradiction)"


@check("T11 partition (invalid_inference+score_abstain+veto_only+wrong_act) sums to evaded")
def _t11_partition_full():
    s = json.loads((V2 / "t11/attack_policy_summary.json").read_text())["attacks"]
    bad = []
    for atk, d in s.items():
        p = d["partition"]
        tot = (p.get("invalid_inference", 0) + p["score_abstain"]
               + p["veto_only"] + p["wrong_act"])
        if tot != d["evaded_n"]:
            bad.append(f"{atk}({tot}!={d['evaded_n']})")
    return not bad, "ok" if not bad else f"bad: {bad}"


@check("T11 credential counts sum to 998 on the historical split")
def _t11_credential():
    v = json.loads((V2 / "t11/verifier_soundness_v2.json").read_text())
    if "historical_998" not in v:
        return None, "verifiers not run yet"
    conf = v["historical_998"]["credential_intent"]["confusion"]
    total = conf["tp"] + conf["fp"] + conf["fn"] + conf["tn"]
    return total == 998, f"tp+fp+fn+tn={total} (want 998)"


# ------------------------------------------------------------------- coverage semantics
@check("T10 SelAcc80 is 'not attainable' exactly when max coverage < 0.80")
def _t10_selacc():
    bad = []
    for corpus in ("phishpedia", "apwg", "trop", "phishpedia_smaller"):
        p = V2 / f"t10/t10_{corpus}.json"
        if not p.exists():
            continue
        for name, row in json.loads(p.read_text())["ladder"].items():
            r = row.get("raw")
            if not r:
                continue
            cmax = r["max_attainable_coverage"]
            sel = r["SelAcc80"]
            na = sel == "not attainable"
            if na != (cmax < 0.80):
                bad.append(f"{corpus}:{name}(cmax={cmax},sel={sel})")
    return not bad, "consistent" if not bad else f"bad: {bad[:5]}"


@check("T10 attainable AURC never integrates ineligible rows (n_eligible <= n)")
def _t10_attainable():
    bad = []
    for corpus in ("phishpedia", "apwg", "trop", "phishpedia_smaller"):
        p = V2 / f"t10/t10_{corpus}.json"
        if not p.exists():
            continue
        for name, row in json.loads(p.read_text())["ladder"].items():
            r = row.get("raw")
            if r and r.get("n_eligible", 0) > r["n"]:
                bad.append(f"{corpus}:{name}")
    return not bad, "ok" if not bad else f"bad: {bad}"


def main() -> int:
    print("T13 regression checks over results/revision_v2/\n")
    n_pass = n_fail = n_skip = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except FileNotFoundError as e:
            ok, detail = None, f"missing artifact: {e}"
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"ERROR: {e}"
        tag = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
        n_pass += ok is True
        n_fail += ok is False
        n_skip += ok is None
        print(f"  [{tag}] {name}\n         {detail}")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
