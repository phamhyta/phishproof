"""Command-line interface of PhishProof."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_panel
from .datasets import read_manifest
from .panel import Panel
from .pipeline import PhishProof
from .policy import DEFAULT_LOGO_THRESHOLD, FrozenPolicy, fit_policy
from .replay import apply_policy, read_evidence, write_evidence
from .schema import PageRecord
from .verification import BrandDomains


def _brand_domains(path: Path | None) -> BrandDomains | None:
    if path is None:
        return None
    return BrandDomains.from_pickle(path) if path.suffix == ".pkl" else BrandDomains.from_json(path)


def _detector(args: argparse.Namespace, policy: FrozenPolicy | None = None) -> PhishProof:
    panel = Panel.from_config(load_panel(args.panel))
    return PhishProof(panel, policy, _brand_domains(args.brand_domains))


def collect(args: argparse.Namespace) -> int:
    detector = _detector(args)
    write_evidence((detector.collect(page) for page in read_manifest(args.manifest)), args.out)
    return 0


def fit(args: argparse.Namespace) -> int:
    groups = json.loads(args.groups.read_text())
    policy = fit_policy(
        list(read_evidence(args.evidence)),
        groups,
        alpha_block=args.alpha_block,
        alpha_clear=args.alpha_clear,
        delta=args.delta,
        logo_threshold=args.logo_threshold,
    )
    policy.save(args.out)
    print(f"tau_block={policy.tau_block} tau_clear={policy.tau_clear}")
    return 0


def replay(args: argparse.Namespace) -> int:
    decisions = apply_policy(read_evidence(args.evidence), FrozenPolicy.load(args.policy))
    with args.out.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), default=lambda value: value.value) + "\n")
    return 0


def score(args: argparse.Namespace) -> int:
    page = PageRecord(
        page_id="page",
        url=args.url,
        dom_html_path=str(args.html),
        screenshot_path=str(args.screenshot),
        raw_dir=str(args.screenshot.parent),
    )
    decision = _detector(args, FrozenPolicy.load(args.policy)).assess(page)
    print(json.dumps(asdict(decision), default=lambda value: value.value, indent=2))
    return 0


def _add_panel_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--panel", type=Path, default=Path("configs/panel.yaml"))
    parser.add_argument(
        "--brand-domains", type=Path, help="reference brand list (.json, or Phishpedia's .pkl)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phishproof", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("score", help="assess one captured page")
    command.add_argument("--url", required=True)
    command.add_argument("--html", required=True, type=Path)
    command.add_argument("--screenshot", required=True, type=Path)
    command.add_argument("--policy", required=True, type=Path)
    _add_panel_options(command)
    command.set_defaults(run=score)

    command = commands.add_parser("collect", help="store per-page evidence for a manifest")
    command.add_argument("--manifest", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    _add_panel_options(command)
    command.set_defaults(run=collect)

    command = commands.add_parser("fit", help="freeze a policy on labelled calibration evidence")
    command.add_argument("--evidence", required=True, type=Path)
    command.add_argument("--groups", required=True, type=Path, help="JSON map of page id to group")
    command.add_argument("--out", required=True, type=Path)
    command.add_argument("--alpha-block", type=float, default=0.02)
    command.add_argument("--alpha-clear", type=float, default=0.02)
    command.add_argument("--delta", type=float, default=0.05)
    command.add_argument("--logo-threshold", type=float, default=DEFAULT_LOGO_THRESHOLD)
    command.set_defaults(run=fit)

    command = commands.add_parser("replay", help="apply a frozen policy to stored evidence")
    command.add_argument("--evidence", required=True, type=Path)
    command.add_argument("--policy", required=True, type=Path)
    command.add_argument("--out", required=True, type=Path)
    command.set_defaults(run=replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)
