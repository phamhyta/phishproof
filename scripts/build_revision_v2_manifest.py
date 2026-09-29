#!/usr/bin/env -S uv run --quiet
"""T13 — assemble the reproducible-package manifest for the revision.

Records, for every revision_v2 artifact and its inputs, the git SHA, file hashes, sizes,
the frozen policy/threshold/calibrator, and the exact commands that regenerate each
output. Licensed captures are LISTED with acquisition instructions, never redistributed.
Emits results/revision_v2/PACKAGE_MANIFEST.json.

Usage
    uv run scripts/build_revision_v2_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

V2 = Path("results/revision_v2")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": path, "present": False}
    return {"path": path, "present": True, "bytes": p.stat().st_size,
            "sha256": sha256(p)}


def main() -> int:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())

    inputs = [
        "results/revision/bundle_or_full.jsonl",
        "results/revision/bundle_apwg_or_full.jsonl",
        "results/revision/bundle_trop_or_full.jsonl",
        "results/bundle_D.jsonl",
        "results/bundle_calib_or.jsonl",
        "results/calibrator_or.json",
        "results/_or/bundle_adversarial_or.jsonl",
        "data/phishsel_final/test.jsonl",
        "data/phishsel_final/calibration.jsonl",
        "data/apwg_final/test.jsonl",
        "data/trop_final/test.jsonl",
        "results/revision/t5_grouped_split.json",
        "artifacts/verifier_soundness.json",
        "configs/panel_or.yaml",
        "configs/panel.yaml",
    ]
    outputs = {
        "T8": ["t8/policy_frozen.json", "t8/logo_calibration_records.jsonl",
               "t8/eligibility_phishpedia.jsonl", "t8/eligibility_apwg.jsonl",
               "t8/eligibility_trop.jsonl", "t8/eligibility_phishpedia_smaller.jsonl",
               "t8/transition_phishpedia.json", "t8/transition_apwg.json",
               "t8/transition_trop.json", "t8/transition_phishpedia_smaller.json"],
        "T9": ["t9/consolidator_phishpedia.jsonl", "t9/consolidator_apwg.jsonl",
               "t9/consolidator_trop.jsonl",
               "t9/consolidator_usage_phishpedia.jsonl",
               "t9/consolidator_usage_apwg.jsonl", "t9/consolidator_usage_trop.jsonl",
               "t9/consolidator_provenance_smaller.jsonl",
               "t9/consolidator_provenance_smaller_summary.json",
               "t9/dawid_skene_disclosure.json"],
        "T10": ["t10/t10_phishpedia.json", "t10/t10_apwg.json", "t10/t10_trop.json",
                "t10/t10_phishpedia_smaller.json", "t10/old_vs_corrected.json"],
        "T11": ["t11/cue_dependence.json", "t11/cue_dependence_records.jsonl",
                "t11/verifier_soundness_v2.json", "t11/verifier_unit_records.jsonl",
                "t11/false_acceptance.json", "t11/attack_policy_records.jsonl",
                "t11/attack_policy_summary.json"],
        "T12": ["t12/summary.json", "t12/timing_records.jsonl",
                "t12/usage_records.jsonl"],
    }
    commands = {
        "T8": ["uv run scripts/revision_t8_policy.py --fit-logo-threshold",
               "uv run scripts/revision_t8_policy.py --corpus <corpus>"],
        "T9": ["uv run scripts/revision_t9_baselines.py dawid-skene",
               "uv run scripts/revision_t9_baselines.py run --corpus <corpus> "
               "  # OpenRouter openai/gpt-4o; OpenAI credits exhausted 2026-09-28",
               "uv run scripts/revision_t9_baselines.py recover-smaller"],
        "T10": ["uv run scripts/revision_t10_metrics.py --corpus <corpus>",
                "uv run scripts/revision_t10_metrics.py --diffs"],
        "T11": ["uv run scripts/revision_t11_evidence.py <subcommand>"],
        "T12": ["uv run scripts/revision_t12_cost.py --n 100 --with-d1"],
        "checks": ["uv run scripts/check_revision_v2.py",
                   "uv run --with pytest pytest tests/test_t8_policy.py"],
    }

    policy = json.loads((V2 / "t8/policy_frozen.json").read_text()) \
        if (V2 / "t8/policy_frozen.json").exists() else {}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha, "git_dirty": dirty,
        "frozen": {"t_logo": policy.get("t_logo"), "tau": policy.get("tau"),
                   "calibrator": policy.get("calibrator"),
                   "calibrator_sha256": policy.get("calibrator_sha256"),
                   "policy_file": str(V2 / "t8/policy_frozen.json")},
        "inputs": [entry(p) for p in inputs],
        "outputs": {task: [entry(str(V2 / f)) for f in files]
                    for task, files in outputs.items()},
        "regeneration_commands": commands,
        "restricted_captures": {
            "note": "APWG (Zenodo 14668190 phishing4190), OpenPhish/Tranco (KnowPhish "
                    "TR-OP), and Phishpedia captures are licensed/restricted and are "
                    "NOT redistributed. Acquisition: APWG phishing4190.zip from Zenodo "
                    "record 14668190; TR-OP.zip from the KnowPhish authors "
                    "(imethanlee/KnowPhish); Phishpedia from lindsey98/Phishpedia. "
                    "Manifests store per-row paths + gold; the input hashes above let a "
                    "holder verify an identical local copy.",
        },
        "provenance_rule": "every output records git SHA, command, config + manifest "
                           "hashes, model snapshot, prompt/schema hash, timestamps, "
                           "provider, and unavailable states in its own meta block.",
    }
    (V2 / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    present = sum(1 for t in manifest["outputs"].values() for e in t if e["present"])
    total = sum(len(t) for t in manifest["outputs"].values())
    miss_in = [e["path"] for e in manifest["inputs"] if not e["present"]]
    print(f"[ok] wrote {V2}/PACKAGE_MANIFEST.json")
    print(f"     outputs present: {present}/{total}; missing inputs: {miss_in or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
