"""Shared evaluation for routing systems."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from banking_router.baselines import KeywordRuleRouter, MajorityQueueRouter, TfidfIntentRouter
from banking_router.data import DatasetRow, load_split


def calculate_metrics(
    true_queues: Sequence[str],
    predicted_queues: Sequence[str],
    reviewed: Sequence[bool] | None = None,
    true_intents: Sequence[str] | None = None,
    predicted_intents: Sequence[str] | None = None,
) -> dict[str, object]:
    if len(true_queues) != len(predicted_queues) or not true_queues:
        raise ValueError("True and predicted queue lists must have the same non-zero length")
    if reviewed is None:
        reviewed = [False] * len(true_queues)
    if len(reviewed) != len(true_queues):
        raise ValueError("Review flags must match the number of predictions")

    total = len(true_queues)
    automatic = [not flag for flag in reviewed]
    automatic_count = sum(automatic)
    wrong_automatic = sum(
        auto and predicted != actual
        for actual, predicted, auto in zip(true_queues, predicted_queues, automatic)
    )
    correct_automatic = automatic_count - wrong_automatic
    raw_correct = sum(actual == predicted for actual, predicted in zip(true_queues, predicted_queues))

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for actual, predicted, is_reviewed in zip(true_queues, predicted_queues, reviewed):
        key = "REVIEW" if is_reviewed else predicted
        confusion[actual][key] += 1

    metrics: dict[str, object] = {
        "rows": total,
        "review_count": total - automatic_count,
        "review_rate": (total - automatic_count) / total,
        "automatic_coverage": automatic_count / total,
        "harmful_wrong_queue_count": wrong_automatic,
        "harmful_wrong_queue_rate": wrong_automatic / total,
        "selective_queue_accuracy": (
            correct_automatic / automatic_count if automatic_count else None
        ),
        "raw_queue_accuracy_without_review": raw_correct / total,
        "queue_confusion": {
            actual: dict(sorted(predictions.items()))
            for actual, predictions in sorted(confusion.items())
        },
    }

    if true_intents is not None and predicted_intents is not None:
        if len(true_intents) != total or len(predicted_intents) != total:
            raise ValueError("Intent labels must match the number of queue predictions")
        metrics["intent_accuracy"] = sum(
            actual == predicted
            for actual, predicted in zip(true_intents, predicted_intents)
        ) / total
    return metrics


def _evaluate_queue_model(
    model_name: str,
    rows: Sequence[DatasetRow],
    predicted_queues: Sequence[str],
    elapsed_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    metrics = calculate_metrics(
        [row.queue for row in rows],
        predicted_queues,
    )
    metrics["elapsed_seconds"] = elapsed_seconds
    records = [
        {
            "model": model_name,
            "text": row.text,
            "true_intent": row.intent,
            "predicted_intent": "",
            "true_queue": row.queue,
            "predicted_queue": predicted,
            "reviewed": False,
        }
        for row, predicted in zip(rows, predicted_queues)
    ]
    return metrics, records


def _evaluate_tfidf(
    model: TfidfIntentRouter,
    rows: Sequence[DatasetRow],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    texts = [row.text for row in rows]
    started = time.perf_counter()
    predicted_intents = model.predict_intents(texts)
    elapsed = time.perf_counter() - started
    from banking_router.taxonomy import queue_for_intent

    predicted_queues = [queue_for_intent(intent) for intent in predicted_intents]
    confidences = model.predict_confidence(texts)
    metrics = calculate_metrics(
        [row.queue for row in rows],
        predicted_queues,
        true_intents=[row.intent for row in rows],
        predicted_intents=predicted_intents,
    )
    metrics["elapsed_seconds"] = elapsed
    metrics["median_confidence"] = sorted(confidences)[len(confidences) // 2]
    records = [
        {
            "model": "tfidf_logistic",
            "text": row.text,
            "true_intent": row.intent,
            "predicted_intent": predicted_intent,
            "true_queue": row.queue,
            "predicted_queue": predicted_queue,
            "confidence": confidence,
            "reviewed": False,
        }
        for row, predicted_intent, predicted_queue, confidence in zip(
            rows, predicted_intents, predicted_queues, confidences
        )
    ]
    return metrics, records


def run_local_evaluation(
    data_dir: Path,
    artifacts_dir: Path,
    results_dir: Path,
    seed: int = 6201,
) -> dict[str, object]:
    train_rows = load_split(data_dir / "train.csv")
    validation_rows = load_split(data_dir / "validation.csv")
    test_rows = load_split(data_dir / "test.csv")

    train_texts = [row.text for row in train_rows]
    train_intents = [row.intent for row in train_rows]
    train_queues = [row.queue for row in train_rows]

    majority = MajorityQueueRouter().fit(train_queues)
    keyword = KeywordRuleRouter().fit(train_queues)
    tfidf = TfidfIntentRouter(seed=seed).fit(train_texts, train_intents)
    tfidf.save(artifacts_dir / "tfidf_logistic.joblib")

    metrics_by_split: dict[str, object] = {}
    all_records: list[dict[str, object]] = []
    for split_name, rows in (("validation", validation_rows), ("test", test_rows)):
        texts = [row.text for row in rows]
        split_metrics: dict[str, object] = {}

        for model_name, model in (("majority_queue", majority), ("keyword_rule", keyword)):
            started = time.perf_counter()
            predictions = model.predict(texts)
            elapsed = time.perf_counter() - started
            model_metrics, records = _evaluate_queue_model(
                model_name, rows, predictions, elapsed
            )
            split_metrics[model_name] = model_metrics
            for record in records:
                record["split"] = split_name
            all_records.extend(records)

        tfidf_metrics, tfidf_records = _evaluate_tfidf(tfidf, rows)
        split_metrics["tfidf_logistic"] = tfidf_metrics
        for record in tfidf_records:
            record["split"] = split_name
        all_records.extend(tfidf_records)
        metrics_by_split[split_name] = split_metrics

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "local_metrics.json"
    metrics_path.write_text(json.dumps(metrics_by_split, indent=2), encoding="utf-8")

    predictions_path = results_dir / "local_predictions.csv"
    fieldnames = [
        "split",
        "model",
        "text",
        "true_intent",
        "predicted_intent",
        "true_queue",
        "predicted_queue",
        "confidence",
        "reviewed",
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})

    return metrics_by_split

