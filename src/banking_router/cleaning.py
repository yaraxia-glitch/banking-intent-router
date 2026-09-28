"""Auditable cleaning and compliance checks for the original BANKING77 files."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I),
    "long_number": re.compile(r"(?<!\d)(?:\d[ -]?){7,19}(?!\d)"),
}

PII_REPLACEMENTS = {
    "email": "[EMAIL]",
    "url": "[URL]",
    "iban": "[IBAN]",
    "long_number": "[LONG_NUMBER]",
}


@dataclass(frozen=True)
class RawRow:
    text: str
    category: str
    source_split: str
    source_row_number: int


@dataclass(frozen=True)
class CleanedRow:
    text: str
    category: str
    source_split: str
    source_row_number: int
    text_sha256: str
    pii_flags: str


@dataclass(frozen=True)
class QuarantineRow:
    source_split: str
    source_row_number: int
    category: str
    text: str
    reason: str
    details: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_text(text: str) -> str:
    """Apply Unicode NFKC, remove control characters, and collapse whitespace."""

    normalised = unicodedata.normalize("NFKC", text)
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalised
    )
    return re.sub(r"\s+", " ", without_controls).strip()


def duplicate_key(text: str) -> str:
    return normalise_text(text).casefold()


def pii_flags(text: str) -> list[str]:
    return [name for name, pattern in PII_PATTERNS.items() if pattern.search(text)]


def redact_pii(text: str) -> tuple[str, list[str]]:
    flags = pii_flags(text)
    redacted = text
    for name in flags:
        redacted = PII_PATTERNS[name].sub(PII_REPLACEMENTS[name], redacted)
    return redacted, flags


def load_categories(path: Path) -> list[str]:
    categories = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise ValueError(f"Expected a JSON list of category names in {path}")
    if len(categories) != 77 or len(set(categories)) != 77:
        raise ValueError(f"Expected 77 unique BANKING77 categories in {path}")
    return categories


def read_raw_csv(path: Path, source_split: str) -> list[RawRow]:
    rows: list[RawRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected BANKING77 columns in {path}: {reader.fieldnames}")
        for row_number, record in enumerate(reader, start=2):
            rows.append(
                RawRow(
                    text=record.get("text", ""),
                    category=record.get("category", ""),
                    source_split=source_split,
                    source_row_number=row_number,
                )
            )
    return rows


def clean_rows(
    train_rows: Sequence[RawRow],
    test_rows: Sequence[RawRow],
    allowed_categories: Sequence[str],
    apply_pii_redaction: bool = True,
) -> tuple[dict[str, list[CleanedRow]], list[QuarantineRow], dict[str, object]]:
    """Clean rows while protecting the official test split from train leakage."""

    allowed = set(allowed_categories)
    quarantine: list[QuarantineRow] = []
    candidates: list[CleanedRow] = []
    audit = Counter()
    pii_counts = Counter()

    for raw in [*train_rows, *test_rows]:
        category = raw.category.strip()
        cleaned_text = normalise_text(raw.text)
        if cleaned_text != raw.text:
            audit["text_normalised"] += 1
        if not cleaned_text:
            quarantine.append(
                QuarantineRow(
                    raw.source_split,
                    raw.source_row_number,
                    category,
                    raw.text,
                    "blank_text",
                    "Text is empty after Unicode and whitespace normalisation",
                )
            )
            continue
        if not category:
            quarantine.append(
                QuarantineRow(
                    raw.source_split,
                    raw.source_row_number,
                    category,
                    raw.text,
                    "blank_category",
                    "Category is empty",
                )
            )
            continue
        if category not in allowed:
            quarantine.append(
                QuarantineRow(
                    raw.source_split,
                    raw.source_row_number,
                    category,
                    raw.text,
                    "invalid_category",
                    "Category is absent from the official categories.json file",
                )
            )
            continue

        flags = pii_flags(cleaned_text)
        for flag in flags:
            pii_counts[flag] += 1
        output_text = cleaned_text
        if apply_pii_redaction and flags:
            output_text, _ = redact_pii(cleaned_text)
            audit["rows_with_pii_redacted"] += 1

        candidates.append(
            CleanedRow(
                text=output_text,
                category=category,
                source_split=raw.source_split,
                source_row_number=raw.source_row_number,
                text_sha256=sha256_text(duplicate_key(output_text)),
                pii_flags=";".join(flags),
            )
        )

    labels_by_key: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        labels_by_key[duplicate_key(row.text)].add(row.category)
    conflicting_keys = {key for key, labels in labels_by_key.items() if len(labels) > 1}

    conflict_free: list[CleanedRow] = []
    for row in candidates:
        key = duplicate_key(row.text)
        if key in conflicting_keys:
            quarantine.append(
                QuarantineRow(
                    row.source_split,
                    row.source_row_number,
                    row.category,
                    row.text,
                    "conflicting_label",
                    "The same normalised text appears with more than one category",
                )
            )
        else:
            conflict_free.append(row)

    deduplicated: list[CleanedRow] = []
    seen_within_split: set[tuple[str, str, str]] = set()
    for row in conflict_free:
        key = (row.source_split, duplicate_key(row.text), row.category)
        if key in seen_within_split:
            quarantine.append(
                QuarantineRow(
                    row.source_split,
                    row.source_row_number,
                    row.category,
                    row.text,
                    "duplicate_within_split",
                    "A previous row in the same source split has identical normalised text and category",
                )
            )
        else:
            seen_within_split.add(key)
            deduplicated.append(row)

    test_keys = {
        duplicate_key(row.text)
        for row in deduplicated
        if row.source_split == "official_test"
    }
    cleaned = {"train": [], "test": []}
    for row in deduplicated:
        key = duplicate_key(row.text)
        if row.source_split == "official_train" and key in test_keys:
            quarantine.append(
                QuarantineRow(
                    row.source_split,
                    row.source_row_number,
                    row.category,
                    row.text,
                    "cross_split_leakage_training_removed",
                    "Normalised text also appears in official_test; test is preserved and train is removed",
                )
            )
            continue
        split_name = "train" if row.source_split == "official_train" else "test"
        cleaned[split_name].append(row)

    for rows in cleaned.values():
        rows.sort(key=lambda row: (row.category, duplicate_key(row.text), row.source_row_number))

    reasons = Counter(row.reason for row in quarantine)
    report: dict[str, object] = {
        "policy": {
            "unicode_normalisation": "NFKC",
            "whitespace": "Control characters replaced with spaces; repeated whitespace collapsed; edges trimmed",
            "empty_rows": "Quarantined",
            "invalid_categories": "Quarantined against official categories.json",
            "conflicting_labels": "All conflicting normalised texts quarantined",
            "within_split_duplicates": "First row retained; later exact normalised duplicates quarantined",
            "cross_split_overlap": "Official test retained; matching official train rows quarantined",
            "pii": "Email, URL, IBAN, and 7-19 digit patterns scanned and redacted when present",
        },
        "raw": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "total_rows": len(train_rows) + len(test_rows),
            "expected_categories": len(allowed),
        },
        "cleaned": {
            "train_rows": len(cleaned["train"]),
            "test_rows": len(cleaned["test"]),
            "total_rows": len(cleaned["train"]) + len(cleaned["test"]),
        },
        "changes": {
            "text_normalised": audit["text_normalised"],
            "rows_with_pii_redacted": audit["rows_with_pii_redacted"],
            "quarantined_total": len(quarantine),
            "quarantine_reasons": dict(sorted(reasons.items())),
            "pii_pattern_matches": dict(sorted(pii_counts.items())),
        },
    }
    return cleaned, quarantine, report


def write_cleaned_csv(rows: Iterable[CleanedRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CleanedRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_quarantine_csv(rows: Iterable[QuarantineRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(QuarantineRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown_report(report: dict[str, object], destination: Path) -> None:
    raw = report["raw"]
    cleaned = report["cleaned"]
    changes = report["changes"]
    reasons = changes["quarantine_reasons"]
    pii_matches = changes["pii_pattern_matches"]

    lines = [
        "# BANKING77 Data Cleaning Report",
        "",
        "## Provenance",
        "",
        "The script downloads the original `train.csv`, `test.csv`, and `categories.json` files directly from the PolyAI-LDN task-specific-datasets repository. Raw files are retained without modification. SHA-256 hashes are recorded in the experiment manifest.",
        "",
        "## Cleaning policy",
        "",
        "- Apply Unicode NFKC normalisation.",
        "- Replace control characters with spaces, collapse repeated whitespace, and trim edges.",
        "- Validate every label against the official 77-category file.",
        "- Quarantine empty rows and conflicting-label records.",
        "- Keep the first normalised duplicate within a split and quarantine later duplicates.",
        "- Preserve the official test row when the same normalised text appears in train and test; quarantine the training copy to prevent leakage.",
        "- Scan for email addresses, URLs, IBAN-like values, and long digit sequences; replace detected values with typed placeholders.",
        "",
        "## Audit results",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Original train rows | {raw['train_rows']:,} |",
        f"| Original test rows | {raw['test_rows']:,} |",
        f"| Cleaned train rows | {cleaned['train_rows']:,} |",
        f"| Cleaned test rows | {cleaned['test_rows']:,} |",
        f"| Text rows changed by normalisation | {changes['text_normalised']:,} |",
        f"| Total quarantined rows | {changes['quarantined_total']:,} |",
        f"| Rows with PII-like values redacted | {changes['rows_with_pii_redacted']:,} |",
        "",
        "### Quarantine reasons",
        "",
    ]
    if reasons:
        lines.extend(f"- `{reason}`: {count}" for reason, count in reasons.items())
    else:
        lines.append("- No rows were quarantined.")
    lines.extend(["", "### PII scan", ""])
    if pii_matches:
        lines.extend(f"- `{name}`: {count}" for name, count in pii_matches.items())
    else:
        lines.append("- No configured PII-like patterns were detected.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `data/cleaned/banking77_train_cleaned.csv`",
            "- `data/cleaned/banking77_test_cleaned.csv`",
            "- `data/cleaned/quarantine.csv`",
            "- `data/cleaned/cleaning_report.json`",
            "",
            "The cleaning process does not prove that the source is free of personal data. It documents the patterns checked and preserves an audit trail for every row removed by the script.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def clean_dataset_files(
    train_path: Path,
    test_path: Path,
    categories_path: Path,
    cleaned_dir: Path,
    report_markdown_path: Path,
    apply_pii_redaction: bool = True,
) -> tuple[dict[str, list[CleanedRow]], dict[str, object]]:
    categories = load_categories(categories_path)
    train_rows = read_raw_csv(train_path, "official_train")
    test_rows = read_raw_csv(test_path, "official_test")
    cleaned, quarantine, report = clean_rows(
        train_rows,
        test_rows,
        categories,
        apply_pii_redaction=apply_pii_redaction,
    )
    train_clean_path = cleaned_dir / "banking77_train_cleaned.csv"
    test_clean_path = cleaned_dir / "banking77_test_cleaned.csv"
    quarantine_path = cleaned_dir / "quarantine.csv"
    write_cleaned_csv(cleaned["train"], train_clean_path)
    write_cleaned_csv(cleaned["test"], test_clean_path)
    write_quarantine_csv(quarantine, quarantine_path)

    report["outputs"] = {
        "train_cleaned": {
            "path": str(train_clean_path),
            "rows": len(cleaned["train"]),
            "sha256": sha256_file(train_clean_path),
        },
        "test_cleaned": {
            "path": str(test_clean_path),
            "rows": len(cleaned["test"]),
            "sha256": sha256_file(test_clean_path),
        },
        "quarantine": {
            "path": str(quarantine_path),
            "rows": len(quarantine),
            "sha256": sha256_file(quarantine_path),
        },
    }
    report_path = cleaned_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, report_markdown_path)
    return cleaned, report
