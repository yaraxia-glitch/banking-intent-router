"""Command-line interface for reproducible project experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banking_router.data import DEFAULT_SEED, prepare_dataset
from banking_router.evaluation import run_local_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="banking-router",
        description="Prepare BANKING77 data and evaluate banking query routing baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Download and sample BANKING77")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--force-download", action="store_true")

    evaluate = subparsers.add_parser("evaluate-local", help="Run all local baselines")
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    evaluate.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    evaluate.add_argument("--results-dir", type=Path, default=Path("results"))
    evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        manifest = prepare_dataset(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            seed=args.seed,
            force_download=args.force_download,
        )
        print(json.dumps(manifest["splits"], indent=2))
    elif args.command == "evaluate-local":
        metrics = run_local_evaluation(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            results_dir=args.results_dir,
            seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

