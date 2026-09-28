"""Command-line interface for reproducible project experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banking_router.cleaning import clean_dataset_files
from banking_router.data import (
    CATEGORIES_URL,
    DEFAULT_SEED,
    TEST_URL,
    TRAIN_URL,
    download_file,
    prepare_dataset,
)
from banking_router.evaluation import run_local_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="banking-router",
        description="Prepare BANKING77 data and evaluate banking query routing baselines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="Download untouched BANKING77 source files")
    download.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--force-download", action="store_true")

    clean = subparsers.add_parser("clean-data", help="Clean and audit the original BANKING77 files")
    clean.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    clean.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    clean.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/data_cleaning_report.md"),
    )
    clean.add_argument(
        "--pii-redaction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    prepare = subparsers.add_parser("prepare-data", help="Download and sample BANKING77")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    prepare.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    prepare.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/data_cleaning_report.md"),
    )
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
    if args.command == "download-data":
        files = {
            "train": download_file(
                TRAIN_URL,
                args.raw_dir / "banking77_train.csv",
                args.force_download,
            ),
            "test": download_file(
                TEST_URL,
                args.raw_dir / "banking77_test.csv",
                args.force_download,
            ),
            "categories": download_file(
                CATEGORIES_URL,
                args.raw_dir / "banking77_categories.json",
                args.force_download,
            ),
        }
        print(json.dumps({name: str(path) for name, path in files.items()}, indent=2))
    elif args.command == "clean-data":
        _, report = clean_dataset_files(
            train_path=args.raw_dir / "banking77_train.csv",
            test_path=args.raw_dir / "banking77_test.csv",
            categories_path=args.raw_dir / "banking77_categories.json",
            cleaned_dir=args.cleaned_dir,
            report_markdown_path=args.report_path,
            apply_pii_redaction=args.pii_redaction,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "prepare-data":
        manifest = prepare_dataset(
            raw_dir=args.raw_dir,
            cleaned_dir=args.cleaned_dir,
            processed_dir=args.processed_dir,
            cleaning_report_path=args.report_path,
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
