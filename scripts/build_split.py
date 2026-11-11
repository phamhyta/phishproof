"""Build the calibration and test manifests of a benchmark from ingested pages."""

import argparse
from pathlib import Path

from phishproof.datasets import (
    BenchmarkSpec,
    build_benchmark,
    credential_login_pages,
    read_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phish", required=True, type=Path)
    parser.add_argument("--benign", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-ordinary", required=True, type=int)
    parser.add_argument("--n-lookalike", required=True, type=int)
    parser.add_argument("--n-benign", required=True, type=int)
    parser.add_argument("--calibration-share", type=float, default=0.15)
    parser.add_argument("--max-per-brand", type=int, default=30)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    spec = BenchmarkSpec(
        n_ordinary=args.n_ordinary,
        n_lookalike=args.n_lookalike,
        n_benign=args.n_benign,
        calibration_share=args.calibration_share,
        max_per_brand=args.max_per_brand,
        seed=args.seed,
    )
    benign = credential_login_pages(read_manifest(args.benign))
    calibration, test = build_benchmark(read_manifest(args.phish), benign, spec)
    write_manifest(calibration, args.out / "calibration.jsonl")
    write_manifest(test, args.out / "test.jsonl")
    print(f"{len(calibration)} calibration and {len(test)} test pages -> {args.out}")


if __name__ == "__main__":
    main()
