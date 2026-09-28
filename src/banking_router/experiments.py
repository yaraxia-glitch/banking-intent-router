"""Reproducible validation and final-test orchestration for Luna and hybrid routing."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Mapping, Sequence

from banking_router.baselines import TfidfIntentRouter
from banking_router.data import DatasetRow, load_split
from banking_router.evaluation import calculate_metrics
from banking_router.hybrid import HybridPolicy, evaluate_policy, select_policy
from banking_router.llm import DEFAULT_MODEL, LLMRoute, OpenAILunaRouter


# Standard short-context rates shown on the official GPT-6 Luna model page on
# 2026-09-28. They are used only for transparent experiment cost estimates.
LUNA_INPUT_USD_PER_MILLION = 0.10
LUNA_OUTPUT_USD_PER_MILLION = 0.50


def row_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_cost_usd(routes: Sequence[LLMRoute]) -> float:
    input_tokens = sum(route.input_tokens for route in routes)
    output_tokens = sum(route.output_tokens for route in routes)
    return (
        input_tokens * LUNA_INPUT_USD_PER_MILLION
        + output_tokens * LUNA_OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def load_llm_cache(path: Path) -> dict[str, LLMRoute]:
    if not path.exists():
        return {}
    cached: dict[str, LLMRoute] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cached[str(record["row_key"])] = LLMRoute.from_dict(record["route"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid cache record at {path}:{line_number}") from exc
    return cached


def _append_cache_record(
    path: Path,
    row: DatasetRow,
    route: LLMRoute,
    latency_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "row_key": row_key(row.text),
        "text": row.text,
        "true_intent": row.intent,
        "true_queue": row.queue,
        "route": route.to_dict(),
        "latency_seconds": latency_seconds,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_llm_routes(
    rows: Sequence[DatasetRow],
    cache_path: Path,
    router: OpenAILunaRouter,
    only_keys: set[str] | None = None,
) -> dict[str, LLMRoute]:
    """Collect missing API predictions and append each result immediately."""

    cached = load_llm_cache(cache_path)
    for row in rows:
        key = row_key(row.text)
        if only_keys is not None and key not in only_keys:
            continue
        if key in cached and cached[key].model == router.model:
            continue
        started = time.perf_counter()
        route = router.route(row.text)
        elapsed = time.perf_counter() - started
        _append_cache_record(cache_path, row, route, elapsed)
        cached[key] = route
    return cached


def _fit_local(data_dir: Path) -> TfidfIntentRouter:
    train_rows = load_split(data_dir / "train.csv")
    return TfidfIntentRouter().fit(
        [row.text for row in train_rows],
        [row.intent for row in train_rows],
    )


def run_luna_validation(
    data_dir: Path,
    results_dir: Path,
    model: str = DEFAULT_MODEL,
    router: OpenAILunaRouter | None = None,
) -> dict[str, object]:
    """Run Luna on validation only; never touches the held-out test split."""

    rows = load_split(data_dir / "validation.csv")
    active_router = router or OpenAILunaRouter.from_environment(model=model)
    cache_path = results_dir / "luna_validation.jsonl"
    routes = collect_llm_routes(rows, cache_path, active_router)
    ordered = [routes[row_key(row.text)] for row in rows]
    metrics = calculate_metrics(
        [row.queue for row in rows],
        [route.queue for route in ordered],
        reviewed=[route.reviewed for route in ordered],
        true_intents=[row.intent for row in rows],
        predicted_intents=[route.intent for route in ordered],
    )
    metrics.update(
        {
            "model": model,
            "input_tokens": sum(route.input_tokens for route in ordered),
            "output_tokens": sum(route.output_tokens for route in ordered),
            "estimated_standard_cost_usd": estimate_cost_usd(ordered),
            "pricing_as_of": "2026-09-28",
        }
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "luna_validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def select_hybrid_policy(
    data_dir: Path,
    results_dir: Path,
    artifacts_dir: Path,
) -> tuple[HybridPolicy, dict[str, object]]:
    """Select and freeze a policy using validation predictions only."""

    rows = load_split(data_dir / "validation.csv")
    local = _fit_local(data_dir)
    texts = [row.text for row in rows]
    local_intents = local.predict_intents(texts)
    local_confidences = local.predict_confidence(texts)
    llm_routes = load_llm_cache(results_dir / "luna_validation.jsonl")
    keys = [row_key(row.text) for row in rows]
    policy, metrics = select_policy(
        rows,
        local_intents,
        local_confidences,
        llm_routes,
        keys,
    )
    policy.save(artifacts_dir / "hybrid_policy.json", metrics)
    return policy, metrics


def _write_final_predictions(
    path: Path,
    rows: Sequence[DatasetRow],
    decisions: Sequence[object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "text",
        "true_intent",
        "predicted_intent",
        "true_queue",
        "predicted_queue",
        "reviewed",
        "source",
        "local_confidence",
        "llm_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row, decision in zip(rows, decisions):
            writer.writerow(
                {
                    "text": row.text,
                    "true_intent": row.intent,
                    "predicted_intent": decision.intent,
                    "true_queue": row.queue,
                    "predicted_queue": decision.queue,
                    "reviewed": decision.reviewed,
                    "source": decision.source,
                    "local_confidence": decision.local_confidence,
                    "llm_confidence": decision.llm_confidence,
                }
            )


def run_final_test(
    data_dir: Path,
    results_dir: Path,
    artifacts_dir: Path,
    model: str = DEFAULT_MODEL,
    router: OpenAILunaRouter | None = None,
) -> dict[str, object]:
    """Evaluate the already-frozen policy once on the official held-out test subset."""

    policy_path = artifacts_dir / "hybrid_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(
            "Freeze a validation-selected policy with select-hybrid-policy first"
        )
    policy = HybridPolicy.load(policy_path)
    rows = load_split(data_dir / "test.csv")
    local = _fit_local(data_dir)
    texts = [row.text for row in rows]
    local_intents = local.predict_intents(texts)
    local_confidences = local.predict_confidence(texts)
    required_keys = {
        row_key(row.text)
        for row, confidence in zip(rows, local_confidences)
        if confidence < policy.local_confidence_threshold
    }

    cache_path = results_dir / "luna_test.jsonl"
    cached = {
        key: route
        for key, route in load_llm_cache(cache_path).items()
        if route.model == model
    }
    missing = required_keys.difference(cached)
    if missing:
        active_router = router or OpenAILunaRouter.from_environment(model=model)
        cached = collect_llm_routes(
            rows,
            cache_path,
            active_router,
            only_keys=required_keys,
        )

    keys = [row_key(row.text) for row in rows]
    metrics, decisions = evaluate_policy(
        rows,
        local_intents,
        local_confidences,
        cached,
        keys,
        policy,
    )
    called_routes = [cached[key] for key in required_keys]
    metrics.update(
        {
            "model": model,
            "input_tokens": sum(route.input_tokens for route in called_routes),
            "output_tokens": sum(route.output_tokens for route in called_routes),
            "estimated_standard_cost_usd": estimate_cost_usd(called_routes),
            "pricing_as_of": "2026-09-28",
            "policy_path": str(policy_path),
        }
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_final_predictions(results_dir / "final_predictions.csv", rows, decisions)
    return metrics
