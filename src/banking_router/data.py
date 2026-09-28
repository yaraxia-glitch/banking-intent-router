"""Download BANKING77 and build deterministic project splits."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from banking_router.taxonomy import SELECTED_INTENTS, queue_for_intent, validate_taxonomy


TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
DEFAULT_SEED = 6201


@dataclass(frozen=True)
class DatasetRow:
    text: str
    intent: str
    queue: str
    source_split: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "banking-intent-router/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def read_source_csv(path: Path, source_split: str) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["text", "category"]:
            raise ValueError(f"Unexpected BANKING77 columns in {path}: {reader.fieldnames}")
        for record in reader:
            intent = record["category"].strip()
            if intent in SELECTED_INTENTS:
                rows.append(
                    DatasetRow(
                        text=record["text"].strip(),
                        intent=intent,
                        queue=queue_for_intent(intent),
                        source_split=source_split,
                    )
                )
    return rows


def _sample_per_intent(
    rows: Sequence[DatasetRow],
    count: int,
    seed: int,
) -> dict[str, list[DatasetRow]]:
    grouped: dict[str, list[DatasetRow]] = defaultdict(list)
    for row in rows:
        grouped[row.intent].append(row)

    sampled: dict[str, list[DatasetRow]] = {}
    for index, intent in enumerate(SELECTED_INTENTS):
        candidates = sorted(grouped[intent], key=lambda item: item.text.casefold())
        if len(candidates) < count:
            raise ValueError(
                f"Intent {intent!r} has {len(candidates)} rows; {count} are required"
            )
        rng = random.Random(seed + index * 1009)
        sampled[intent] = rng.sample(candidates, count)
    return sampled


def build_splits(
    train_rows: Sequence[DatasetRow],
    test_rows: Sequence[DatasetRow],
    seed: int = DEFAULT_SEED,
) -> dict[str, list[DatasetRow]]:
    """Build 240 train, 60 validation, and 240 test rows."""

    validate_taxonomy()
    selected_train = _sample_per_intent(train_rows, count=25, seed=seed)
    selected_test = _sample_per_intent(test_rows, count=20, seed=seed + 7919)

    splits: dict[str, list[DatasetRow]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for intent in SELECTED_INTENTS:
        splits["train"].extend(selected_train[intent][:20])
        splits["validation"].extend(selected_train[intent][20:25])
        splits["test"].extend(selected_test[intent])

    for offset, split_name in enumerate(("train", "validation", "test")):
        random.Random(seed + offset).shuffle(splits[split_name])
    return splits


def write_split(rows: Iterable[DatasetRow], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["text", "intent", "queue", "source_split"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_split(path: Path) -> list[DatasetRow]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["text", "intent", "queue", "source_split"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected processed columns in {path}: {reader.fieldnames}")
        return [DatasetRow(**record) for record in reader]


def _split_counts(rows: Sequence[DatasetRow]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "intents": dict(sorted(Counter(row.intent for row in rows).items())),
        "queues": dict(sorted(Counter(row.queue for row in rows).items())),
    }


def prepare_dataset(
    raw_dir: Path,
    processed_dir: Path,
    seed: int = DEFAULT_SEED,
    force_download: bool = False,
) -> dict[str, object]:
    train_path = download_file(TRAIN_URL, raw_dir / "banking77_train.csv", force_download)
    test_path = download_file(TEST_URL, raw_dir / "banking77_test.csv", force_download)
    train_rows = read_source_csv(train_path, "official_train")
    test_rows = read_source_csv(test_path, "official_test")
    splits = build_splits(train_rows, test_rows, seed)

    output_paths: dict[str, Path] = {}
    for split_name, rows in splits.items():
        path = processed_dir / f"{split_name}.csv"
        write_split(rows, path)
        output_paths[split_name] = path

    manifest = {
        "dataset": "BANKING77",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
        "license": "CC BY 4.0",
        "seed": seed,
        "selected_intents": list(SELECTED_INTENTS),
        "source_files": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "test": {"path": str(test_path), "sha256": sha256_file(test_path)},
        },
        "splits": {
            name: {
                **_split_counts(rows),
                "path": str(output_paths[name]),
                "sha256": sha256_file(output_paths[name]),
            }
            for name, rows in splits.items()
        },
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

