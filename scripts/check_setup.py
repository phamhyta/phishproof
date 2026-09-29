"""Phase 0 smoke test: schema, configs, and cache all load and round-trip.

Run:  .venv/bin/python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phishproof.cache import JsonCache
from phishproof.config import load_experiment, load_panel
from phishproof.schema import (
    ACTIVE_CUE_TYPES,
    AgentOutput,
    Cue,
    CueType,
    GEAResult,
    Label,
    PageRecord,
)


def main() -> int:
    ok = True

    # 1) Schema: build a sample page + two agent outputs with typed cues.
    page = PageRecord(
        page_id="demo-0001",
        url="http://paypal-secure.com/login",
        label=Label.PHISH,
        screenshot_path="data/phishsel/demo-0001/shot.png",
        dom_html_path="data/phishsel/demo-0001/dom.html",
        brand="PayPal",
        source="phishpedia",
        split="test",
    )
    a1 = AgentOutput(
        agent_id="agent_a_text",
        verdict=Label.PHISH,
        cues=[
            Cue(type=CueType.BRAND_CLAIM, value="paypal", raw_value="PayPal",
                asserted_by="agent_a_text"),
            Cue(type=CueType.FORM_ACTION_DOMAIN, value="paypal-secure.com",
                raw_value="https://paypal-secure.com/signin", asserted_by="agent_a_text"),
        ],
    )
    a2 = AgentOutput(
        agent_id="agent_c_vision",
        verdict=Label.PHISH,
        cues=[Cue(type=CueType.LOGO_BRAND, value="paypal", asserted_by="agent_c_vision")],
    )
    assert a1.cues[0].key() == ("brand_claim", "paypal")
    print(f"[ok] schema: page={page.page_id} agents={a1.agent_id},{a2.agent_id} "
          f"cues={len(a1.cues) + len(a2.cues)}")
    print(f"[ok] active cue types: {[c.value for c in ACTIVE_CUE_TYPES]}")

    # 2) A GEAResult round-trips through JSON.
    res = GEAResult(page_id=page.page_id, verdict=Label.PHISH, agreement=0.75,
                    groundedness=0.9, gea=0.675, consensus_cues=a1.cues, acted=True)
    assert GEAResult.model_validate_json(res.model_dump_json()).gea == 0.675
    print("[ok] GEAResult JSON round-trip")

    # 3) Configs load.
    try:
        panel = load_panel()
        exp = load_experiment()
        print(f"[ok] panel: {[a.id + '(' + a.model + ')' for a in panel.panel]}")
        print(f"[ok] spot_check: {panel.spot_check.model} on {panel.spot_check.n_pages} pages")
        print(f"[ok] experiment: n_pages={exp.n_pages} seeds={exp.seeds}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] config load: {e}")

    # 4) Cache round-trips and dedupes.
    cache = JsonCache("data/cache")
    cache.set("llama3.1:8b", "hello prompt", {"verdict": "phish"})
    got = cache.get("llama3.1:8b", "hello prompt")
    assert got == {"verdict": "phish"}, got
    assert cache.get("llama3.1:8b", "different prompt") is None
    print("[ok] cache set/get + miss")

    print("\nPhase 0 OK" if ok else "\nPhase 0 had failures")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
