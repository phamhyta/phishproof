# PhishProof

PhishProof is a trustworthy, explainable phishing detector: a cross-modal
multi-agent LLM panel cites typed evidence, scores how strongly the agents agree
per evidence type (the Grounded Evidence-Agreement score), verifies each agreed
cue against the page, and abstains when the evidence is not trustworthy enough
to act on.

This repository is the implementation of our paper, which is currently under review. The full executable source code, data, and artifacts will released here once the paper is accepted.

## Pipeline

A page flows through five stages (`phishproof/pipeline.py`):

1. **Context** (`agents/page_context.py`) -- assemble the screenshot, DOM, URL.
2. **Panel** (`agents/`) -- a vision agent and text agents cite typed cues.
3. **Aggregate + ground** (`aggregate/`, `tools/`) -- per-type agreement (GEA)
   and re-derive each agreed cue from the page.
4. **Calibrate** (`calibration/`) -- map the trust signal to P(correct).
5. **Decide** (`calibration/selective.py`) -- act above the operating threshold,
   else abstain; return the grounded cues as the explanation.

## Package map

| Module | Role |
|--------|------|
| `phishproof.schema` | Typed evidence cue schema and data models |
| `phishproof.agents` | Cross-modal, tool-using agent panel |
| `phishproof.aggregate` | Per-type agreement and the GEA trust signal |
| `phishproof.tools` | Grounding verifiers (brand, DOM, logo, cert, redirect) |
| `phishproof.calibration` | Isotonic calibration and the selective rule |
| `phishproof.baselines` | Reliability baselines compared in the paper |
| `phishproof.eval` | Selective-prediction metrics and the adversarial study |
| `scripts/` | Corpus ingestion, panel run, and evaluation entry points |

## Input / output contract

Each input line is one JSON object:

```json
{"id": "sample-1", "url": "https://login.example.test/account", "title": "Account sign-in"}
```

Fields: `id`, `url`, `title?`, `html_path?`, `screenshot_path?`, `metadata?`.
Each output line is one JSON object with `id`, `label`, `confidence`,
`decision` (`act` / `abstain`), and `reasons` (the grounded evidence cues).

## Install

```bash
uv sync
```

The package imports with no third-party dependencies; the released
implementation adds the stack listed under the `full` extra in `pyproject.toml`.

## Release status

This is a pre-acceptance public skeleton. The private research repository keeps
the implementation, manuscript, experiment scripts, result bundles, and data
manifests until the review process is complete.

## License

MIT. See `LICENSE`.
