"""B5 — reliability-weighted label agreement via Dawid-Skene EM (dawid1979maximum).

Treats each agent as an annotator and each page as an item; estimates per-agent
confusion matrices + class priors, and returns each page's posterior confidence in the
inferred class. This is a *label*-level reliability score — the baseline PhishProof beats
by scoring agreement on evidence instead (RQ1, ablation).
"""

from __future__ import annotations

import numpy as np

from ..schema import AgentOutput, Label

_L = {Label.BENIGN: 0, Label.PHISH: 1}


def _observations(outputs_by_page: dict[str, list[AgentOutput]]):
    page_ids = list(outputs_by_page)
    workers = sorted({o.agent_id for outs in outputs_by_page.values() for o in outs})
    widx = {w: i for i, w in enumerate(workers)}
    obs = []
    for pid in page_ids:
        obs.append({widx[o.agent_id]: _L[o.verdict] for o in outputs_by_page[pid]})
    return page_ids, workers, obs


def dawid_skene(outputs_by_page: dict[str, list[AgentOutput]], n_iter: int = 50):
    page_ids, workers, obs = _observations(outputs_by_page)
    n_items, n_classes, n_workers = len(obs), 2, len(workers)

    # init item posteriors by vote share
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
        conf = np.full((n_workers, n_classes, n_classes), 1e-9)  # [worker][true][observed]
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

    return {pid: float(T[i].max()) for i, pid in enumerate(page_ids)}
