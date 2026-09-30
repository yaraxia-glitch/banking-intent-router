"""Read-only verification of data provenance, experiment outputs, and tests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from banking_router.cleaning import duplicate_key  # noqa: E402


class Verification:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        marker = "PASS" if condition else "FAIL"
        print(f"[{marker}] {message}")
        if not condition:
            self.failures.append(message)


def load_json(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def jsonl_keys(path: Path) -> list[str]:
    keys: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                keys.append(str(json.loads(line)["row_key"]))
    return keys


def processed_text_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {duplicate_key(row["text"]) for row in csv.DictReader(stream)}


def verify_hashes_and_counts(checks: Verification) -> None:
    manifest = load_json("data/processed/manifest.json")
    checks.check(manifest["dataset"] == "BANKING77", "Dataset is BANKING77")
    checks.check(manifest["license"] == "CC BY 4.0", "Dataset license is recorded")
    checks.check(manifest["seed"] == 6201, "Sampling seed is fixed at 6201")

    for label, record in manifest["source_files"].items():
        path = ROOT / record["path"]
        checks.check(path.exists(), f"Raw source file exists: {label}")
        if path.exists():
            checks.check(
                sha256_file(path) == record["sha256"],
                f"Raw source hash matches manifest: {label}",
            )

    for label, record in manifest["cleaning"]["outputs"].items():
        path = ROOT / record["path"]
        checks.check(path.exists(), f"Cleaning output exists: {label}")
        if path.exists():
            checks.check(
                sha256_file(path) == record["sha256"],
                f"Cleaning output hash matches manifest: {label}",
            )
            checks.check(
                csv_row_count(path) == record["rows"],
                f"Cleaning output row count matches: {label}",
            )

    expected_rows = {"train": 240, "validation": 60, "test": 240}
    split_keys: dict[str, set[str]] = {}
    for split, expected in expected_rows.items():
        record = manifest["splits"][split]
        path = ROOT / record["path"]
        checks.check(path.exists(), f"Processed split exists: {split}")
        if path.exists():
            checks.check(csv_row_count(path) == expected, f"{split} contains {expected} rows")
            checks.check(
                sha256_file(path) == record["sha256"],
                f"Processed split hash matches manifest: {split}",
            )
            split_keys[split] = processed_text_keys(path)

    if set(split_keys) == set(expected_rows):
        checks.check(
            not split_keys["train"].intersection(split_keys["validation"]),
            "Train and validation texts do not overlap",
        )
        checks.check(
            not split_keys["train"].intersection(split_keys["test"]),
            "Train and test texts do not overlap",
        )
        checks.check(
            not split_keys["validation"].intersection(split_keys["test"]),
            "Validation and test texts do not overlap",
        )


def verify_experiments(checks: Verification) -> None:
    local = load_json("results/local_metrics.json")
    validation = load_json("results/luna_validation_metrics.json")
    final = load_json("results/final_metrics.json")
    policy = load_json("artifacts/hybrid_policy.json")

    checks.check(
        local["test"]["tfidf_logistic"]["harmful_wrong_queue_rate"] == 0.05,
        "Local TF-IDF test harmful wrong-queue rate is 5.0%",
    )
    checks.check(validation["rows"] == 60, "Luna validation contains 60 rows")
    checks.check(validation["gateway"] == "openrouter", "Validation used OpenRouter")
    checks.check(
        validation["model"] == "openai/gpt-6-luna",
        "Validation used openai/gpt-6-luna",
    )

    validation_keys = jsonl_keys(ROOT / "results/luna_validation.jsonl")
    test_keys = jsonl_keys(ROOT / "results/luna_test.jsonl")
    checks.check(len(validation_keys) == 60, "Validation cache contains 60 API responses")
    checks.check(len(set(validation_keys)) == 60, "Validation cache keys are unique")
    checks.check(len(test_keys) == 9, "Test cache contains only 9 API responses")
    checks.check(len(set(test_keys)) == 9, "Test cache keys are unique")

    checks.check(policy["policy"]["selected_on"] == "validation", "Policy was selected on validation")
    checks.check(
        policy["selection_metrics"]["constraints_met"] is True,
        "Frozen validation policy met both constraints",
    )
    checks.check(final["rows"] == 240, "Final evaluation contains 240 test rows")
    checks.check(final["gateway"] == "openrouter", "Final test used OpenRouter")
    checks.check(
        final["model"] == "openai/gpt-6-luna",
        "Final test used openai/gpt-6-luna",
    )
    checks.check(
        final["harmful_wrong_queue_rate"] <= 0.05,
        "Final harmful wrong-queue rate is at most 5%",
    )
    checks.check(final["review_rate"] <= 0.10, "Final review rate is at most 10%")
    checks.check(final["api_call_count"] == 9, "Final test made 9 rented-model calls")
    checks.check(
        csv_row_count(ROOT / "results/final_predictions.csv") == 240,
        "Final predictions contain 240 rows",
    )


def verify_no_embedded_secret(checks: Verification) -> None:
    secret_pattern = re.compile(r"sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}")
    excluded = {".git", ".venv", "data", "results"}
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.relative_to(ROOT).parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secret_pattern.search(content):
            findings.append(str(path.relative_to(ROOT)))
    checks.check(not findings, "No API key is embedded in project source or documentation")


def run_tests(checks: Verification) -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    checks.check(result.wasSuccessful(), f"All {result.testsRun} automated tests pass")


def main() -> int:
    checks = Verification()
    print("Verifying PE6201 banking intent router\n")
    verify_hashes_and_counts(checks)
    verify_experiments(checks)
    verify_no_embedded_secret(checks)
    run_tests(checks)
    print()
    if checks.failures:
        print(f"VERIFICATION FAILED: {len(checks.failures)} check(s) failed")
        return 1
    print("VERIFICATION PASSED: data, outputs, constraints, secrets, and tests are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
