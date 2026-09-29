"""Phase 3 smoke test: panel + aggregation + GEA on real pages, using mock agents.

Demonstrates the three regimes the method must distinguish, with NO model required:
  (A) high-GEA   — agents converge on the same cues, all grounded -> act
  (B) disjoint   — agents cite different cues, no majority -> A collapses -> abstain  (RLWR)
  (C) hallucination — agents agree on a cue the page does not carry -> G collapses -> abstain

Run:  .venv/bin/python scripts/check_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.aggregate.gea import score_page
from phishproof.data_io import read_manifest
from phishproof.schema import CueType, Label
from phishproof.tools.dom import form_action_domains
from phishproof.tools.registry import GroundingContext


class FakeEmbedder:
    def similarity(self, crop, brand: str) -> float:
        return 0.9  # stand-in for CLIP; real model in Phase 3 run


def show(name: str, outputs, page, ctx) -> None:
    res = score_page(outputs, page, ctx)
    cons = ", ".join(f"{c.type.value}={c.value}" for c in res.consensus_cues) or "(none)"
    print(f"  [{name}] verdict={res.verdict.value}  A={res.agreement:.2f}  "
          f"G={res.groundedness:.2f}  GEA={res.gea:.3f}")
    print(f"      consensus: {cons}")


def main() -> int:
    manifest = Path("data/phishsel/manifest_phish.jsonl")
    if not manifest.exists():
        print("[FAIL] run ingest_phishpedia.py first")
        return 1

    # pick a page that has a form-action domain so grounding is meaningful
    page = next(r for r in read_manifest(manifest)[:50] if form_action_domains(r))
    ctx = GroundingContext(logo_embedder=FakeEmbedder())
    print(f"Page {page.page_id}  brand={page.brand!r}  "
          f"form_domain={sorted(form_action_domains(page))[0]}\n")

    from phishproof.agents.mock import MockAgent

    # (A) three agents agree on the real cues
    agree = [MockAgent(f"m{i}") for i in range(3)]
    show("A high-GEA ", [a.analyze(page) for a in agree], page, ctx)

    # (B) disjoint: each cites a different form-action domain, drops the rest
    others = {CueType.BRAND_CLAIM, CueType.CREDENTIAL_INTENT, CueType.LOGO_BRAND}
    disjoint = [
        MockAgent("m0", drop=others, override={CueType.FORM_ACTION_DOMAIN: "alpha-x.com"}),
        MockAgent("m1", drop=others, override={CueType.FORM_ACTION_DOMAIN: "beta-y.com"}),
        MockAgent("m2", drop=others, override={CueType.FORM_ACTION_DOMAIN: "gamma-z.com"}),
    ]
    show("B disjoint ", [a.analyze(page) for a in disjoint], page, ctx)

    # (C) shared hallucination: all agree on a form-action domain the page lacks
    hallu = [
        MockAgent(f"m{i}", drop=others,
                  override={CueType.FORM_ACTION_DOMAIN: "not-on-this-page.com"})
        for i in range(3)
    ]
    show("C hallucin.", [a.analyze(page) for a in hallu], page, ctx)

    print("\nPhase 3 panel + aggregation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
