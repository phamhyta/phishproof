# PhishProof

Implementation of *Selective Phishing Detection with Verified Evidence across Language and Vision Models*.

Three separately queried models (two text, one vision) each return a verdict, a confidence and typed evidence cues for a captured web page. PhishProof ranks the page by the mean of the three confidences, re-derives the cues the panel agrees on from the capture, and then blocks, clears or sends the page to a person.

A page is acted on only if:

- every member returned a confidence;
- the consensus contains a form-action or credential cue, and every such cue passes its check;
- the score reaches the threshold of the action.

A clear also needs brand-host binding: every brand the panel names has to match the page's registrable label, and a brand on the reference list has to be served from one of its listed domains.

## Layout

```
phishproof/
  schema/        records passed between layers: cues, page records, page evidence
  panel/         model client, fixed prompt, response parsing, the panel itself
  evidence/      cue normalization, strict-majority consensus, the GEA diagnostic
  verification/  checks on cited cues (DOM, brand, logo) and brand-host binding
  policy/        ranking score, threshold scan, frozen policy, act-or-abstain rule
  calibration/   isotonic map that reports a correctness estimate
  datasets/      corpus loaders, manifests, splits, near-duplicate groups
  metrics/       tie-averaged risk-coverage metrics, bootstrap intervals
  baselines/     reliability baselines B1-B5 and the dedicated detectors D1-D5
  pipeline.py    ties the layers together for one page
  replay.py      applies a frozen policy to stored evidence
  cli.py         command-line entry point
scripts/         build manifests from the source corpora
configs/         the model panel
reproducibility/ prompts and response schema
```

Dependencies point downwards: `policy` reads `verification` and `evidence`, which read `schema`. Nothing below `pipeline.py` knows about the command line.

## Install

```bash
uv venv
uv pip install -e .            # add ".[logo]" for the CLIP logo check
cp .env.example .env           # then fill in your API keys
```

## Use

From Python:

```python
from phishproof import FrozenPolicy, PageRecord, PhishProof
from phishproof.panel import Panel

detector = PhishProof(Panel.from_config(), FrozenPolicy.load("policy.json"))
page = PageRecord(page_id="p1", url="https://example.com/login",
                  dom_html_path="html.txt", screenshot_path="shot.png")
print(detector.assess(page))
```

From the command line:

```bash
phishproof score --url https://example.com/login --html html.txt --screenshot shot.png --policy policy.json
```

The thresholds are fitted once on labelled calibration pages, then frozen:

```bash
phishproof collect --manifest calibration.jsonl --out evidence.jsonl
phishproof fit --evidence evidence.jsonl --groups groups.json --out policy.json
phishproof replay --evidence other.jsonl --policy policy.json --out decisions.jsonl
```

`replay` makes no model calls. In Python, the same step is `phishproof.apply_policy`.

D1-D3 wrap the released Phishpedia, PhishIntention and PhishLLM code, which you install separately. D4 (KnowPhish) and D5 (PhishAgent) are re-implementations; install the `detectors` and `retriever` extras to use them.

Hosted models change over time, so a new run may not match stored responses exactly. Binding uses the reference brand list released with Phishpedia (`domain_map.pkl`); pass it with `--brand-domains`. It is a pickle file, so load only a copy you trust.

## Data

Page captures are not redistributed. Get them from the source corpora under their own terms, then build manifests with `scripts/`:

| Corpus | Source | Script |
|--------|--------|--------|
| Phishpedia | [lindsey98/Phishpedia](https://github.com/lindsey98/Phishpedia) | `ingest_phishpedia.py` |
| APWG `phishing4190` | [Zenodo 14668190](https://zenodo.org/records/14668190) | `ingest_apwg.py` |
| KnowPhish TR-OP `openphish_5000` | KnowPhish authors | `ingest_trop.py` |

`build_split.py` mixes ordinary and lookalike phishing pages (those whose domain contains the brand) with benign pages, then writes the calibration and test manifests. `find_duplicates.py` writes the group map that `phishproof fit` reads. Remove test duplicates from the calibration split first, with `phishproof.datasets.without_test_duplicates`.

## License

MIT (see `LICENSE`). Page captures remain under the terms of their source corpora.
