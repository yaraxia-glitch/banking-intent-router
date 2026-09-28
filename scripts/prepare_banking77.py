"""Download, clean, audit, and sample BANKING77 in one reproducible command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banking_router.data import DEFAULT_SEED, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cleaned-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/data_cleaning_report.md"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    manifest = prepare_dataset(
        raw_dir=args.raw_dir,
        cleaned_dir=args.cleaned_dir,
        processed_dir=args.processed_dir,
        cleaning_report_path=args.report_path,
        seed=args.seed,
        force_download=args.force_download,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

