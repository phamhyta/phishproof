#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Fixed-calibrator transfer for tab_brands_out, emitted to a committed artifact so the
table's cells are traceable. Applies the frozen Phishpedia calibrator (already baked into
each eligibility row's trust_calibrated) and computes, over attainable coverage, the
deployed score's AURC (deterministic page-id ties), SelAcc80 where attainable, and ECE on
the calibrated trust, with 2000-resample random-tie intervals. Writes
results/revision_v2/t10/transfer_70b.json.

Usage: uv run scripts/revision_transfer_70b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CORPORA = {"phishpedia": "eligibility_phishpedia.jsonl",
           "trop": "eligibility_trop.jsonl",
           "apwg": "eligibility_apwg.jsonl"}
T8 = Path("results/revision_v2/t8")
OUT = Path("results/revision_v2/t10/transfer_70b.json")


def main():
    rng = np.random.RandomState(0)
    out = {"convention": "frozen Phishpedia calibrator; attainable-coverage AURC "
                         "(deterministic page-id ties); ECE on calibrated trust; "
                         "2000 random-tie page resamples", "corpora": {}}
    for c, f in CORPORA.items():
        rows = [json.loads(l) for l in (T8 / f).read_text().splitlines() if l.strip()]
        n = len(rows)
        pids = np.array([r["page_id"] for r in rows])
        pr = np.argsort(np.argsort(pids)).astype(float)
        s = np.array([r["s_raw"] if r["s_raw"] is not None else np.nan for r in rows])
        tr = np.array([r["trust_calibrated"] if r["trust_calibrated"] is not None else np.nan
                       for r in rows])
        cor = np.array([r["verdict"] == r["label"] if r["verdict"] else False for r in rows], float)
        valid = np.array([r["complete"]["valid_inference"] for r in rows])
        elig = valid & ~np.isnan(s)
        k = int(elig.sum()); cmax = k / n
        ei = np.where(elig)[0]
        o = ei[np.lexsort((pr[ei], -s[ei]))]
        A = 100 * float(np.mean(np.cumsum(1 - cor[o]) / np.arange(1, len(o) + 1)))
        S = (100 * float(np.mean(cor[o[:int(round(0.8 * n))]]))) if cmax >= 0.8 else None
        pp = np.clip(tr[ei][~np.isnan(tr[ei])], 0, 1); cc = cor[ei][~np.isnan(tr[ei])]
        edges = np.linspace(0, 1, 11); E = 0.0
        for i in range(10):
            hi = i == 9
            m = (pp >= edges[i]) & ((pp <= edges[i + 1]) if hi else (pp < edges[i + 1]))
            if m.sum():
                E += m.sum() / len(pp) * abs(cc[m].mean() - pp[m].mean())
        Ab, Sb, Eb = [], [], []
        for _ in range(2000):
            bi = rng.randint(0, n, n); m = elig[bi] & ~np.isnan(s[bi]); sub = bi[m]
            if len(sub) < 5:
                continue
            jit = rng.permutation(len(sub)).astype(float)
            oo = np.lexsort((jit, -s[sub]))
            Ab.append(100 * float(np.mean(np.cumsum(1 - cor[sub][oo]) / np.arange(1, len(oo) + 1))))
            if cmax >= 0.8:
                kk = max(1, int(round(0.8 * len(bi))))
                if len(oo) >= kk:
                    Sb.append(100 * float(np.mean(cor[sub][oo[:kk]])))
            trs = tr[sub]; mm = ~np.isnan(trs); ppb = np.clip(trs[mm], 0, 1); ccb = cor[sub][mm]
            e = 0.0
            for i in range(10):
                hi = i == 9
                q = (ppb >= edges[i]) & ((ppb <= edges[i + 1]) if hi else (ppb < edges[i + 1]))
                if q.sum():
                    e += q.sum() / len(ppb) * abs(ccb[q].mean() - ppb[q].mean())
            Eb.append(e)
        ci = lambda b, d: [round(float(np.percentile(b, 2.5)), d), round(float(np.percentile(b, 97.5)), d)]
        out["corpora"][c] = {"n_eligible": k, "cmax": round(cmax, 3),
                             "AURC": round(A, 2), "AURC_ci": ci(Ab, 2),
                             "SelAcc80": None if S is None else round(S, 1),
                             "SelAcc80_ci": ci(Sb, 1) if Sb else None,
                             "ECE": round(E, 3), "ECE_ci": ci(Eb, 3)}
        print(f"{c}: AURC {out['corpora'][c]['AURC']} {out['corpora'][c]['AURC_ci']} "
              f"SelAcc {out['corpora'][c]['SelAcc80']} ECE {out['corpora'][c]['ECE']}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    main()
