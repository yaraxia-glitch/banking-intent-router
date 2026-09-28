"""Validation-selected hybrid policy for local and rented model predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from banking_router.data import DatasetRow
from banking_router.evaluation import calculate_metrics
from banking_router.llm import LLMRoute, UNKNOWN_INTENT
from banking_router.taxonomy import queue_for_intent


@dataclass(frozen=True)
class HybridPolicy:
    local_confidence_threshold: float
    llm_confidence_threshold: float
    review_on_queue_disagreement: bool = True
    selected_on: str = "validation"
    target_harmful_wrong_queue_rate: float = 0.05
    maximum_review_rate: float = 0.10

    def save(self, path: Path, selection_metrics: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"policy": asdict(self), "selection_metrics": selection_metrics}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HybridPolicy":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload["policy"])


@dataclass(frozen=True)
class HybridDecision:
    intent: str
    queue: str
    reviewed: bool
    source: str
    local_confidence: float
    llm_confidence: float | None


def apply_policy(
    local_intent: str,
    local_confidence: float,
    llm_route: LLMRoute | None,
    policy: HybridPolicy,
) -> HybridDecision:
    local_queue = queue_for_intent(local_intent)
    if local_confidence >= policy.local_confidence_threshold:
        return HybridDecision(
            intent=local_intent,
            queue=local_queue,
            reviewed=False,
            source="local",
            local_confidence=local_confidence,
            llm_confidence=None,
        )
    if llm_route is None:
        raise ValueError("An LLM result is required below the local confidence threshold")

    if (
        llm_route.reviewed
        or llm_route.intent == UNKNOWN_INTENT
        or llm_route.confidence < policy.llm_confidence_threshold
    ):
        return HybridDecision(
            intent=local_intent,
            queue=local_queue,
            reviewed=True,
            source="review_low_llm_confidence",
            local_confidence=local_confidence,
            llm_confidence=llm_route.confidence,
        )

    llm_queue = queue_for_intent(llm_route.intent)
    if policy.review_on_queue_disagreement and llm_queue != local_queue:
        return HybridDecision(
            intent=llm_route.intent,
            queue=llm_queue,
            reviewed=True,
            source="review_queue_disagreement",
            local_confidence=local_confidence,
            llm_confidence=llm_route.confidence,
        )
    return HybridDecision(
        intent=llm_route.intent,
        queue=llm_queue,
        reviewed=False,
        source="llm",
        local_confidence=local_confidence,
        llm_confidence=llm_route.confidence,
    )


def _evaluate_candidate(
    rows: Sequence[DatasetRow],
    local_intents: Sequence[str],
    local_confidences: Sequence[float],
    llm_routes: Mapping[str, LLMRoute],
    policy: HybridPolicy,
    row_keys: Sequence[str],
) -> tuple[dict[str, object], list[HybridDecision]]:
    decisions = [
        apply_policy(
            local_intent,
            confidence,
            llm_routes.get(key),
            policy,
        )
        for local_intent, confidence, key in zip(
            local_intents, local_confidences, row_keys
        )
    ]
    metrics = calculate_metrics(
        [row.queue for row in rows],
        [decision.queue for decision in decisions],
        reviewed=[decision.reviewed for decision in decisions],
        true_intents=[row.intent for row in rows],
        predicted_intents=[decision.intent for decision in decisions],
    )
    metrics["api_call_count"] = sum(
        confidence < policy.local_confidence_threshold
        for confidence in local_confidences
    )
    metrics["api_call_rate"] = metrics["api_call_count"] / len(rows)
    return metrics, decisions


def select_policy(
    rows: Sequence[DatasetRow],
    local_intents: Sequence[str],
    local_confidences: Sequence[float],
    llm_routes: Mapping[str, LLMRoute],
    row_keys: Sequence[str],
    target_wrong_rate: float = 0.05,
    maximum_review_rate: float = 0.10,
) -> tuple[HybridPolicy, dict[str, object]]:
    """Choose thresholds using validation data only.

    Feasible candidates meet both project constraints. Among them, selection
    minimizes rented-model calls, then harmful errors, then review. If no candidate
    is feasible, the safest candidate is returned with ``constraints_met=false``.
    """

    if not rows or not (
        len(rows)
        == len(local_intents)
        == len(local_confidences)
        == len(row_keys)
    ):
        raise ValueError("Validation inputs must have the same non-zero length")
    missing = [key for key in row_keys if key not in llm_routes]
    if missing:
        raise ValueError(f"Missing {len(missing)} validation LLM predictions")

    local_thresholds = sorted({0.0, 1.000001, *local_confidences})
    llm_thresholds = sorted(
        {0.0, 1.000001, *(route.confidence for route in llm_routes.values())}
    )
    candidates: list[tuple[HybridPolicy, dict[str, object]]] = []
    for local_threshold in local_thresholds:
        for llm_threshold in llm_thresholds:
            policy = HybridPolicy(
                local_confidence_threshold=local_threshold,
                llm_confidence_threshold=llm_threshold,
                target_harmful_wrong_queue_rate=target_wrong_rate,
                maximum_review_rate=maximum_review_rate,
            )
            metrics, _ = _evaluate_candidate(
                rows,
                local_intents,
                local_confidences,
                llm_routes,
                policy,
                row_keys,
            )
            metrics["constraints_met"] = (
                metrics["harmful_wrong_queue_rate"] <= target_wrong_rate
                and metrics["review_rate"] <= maximum_review_rate
            )
            candidates.append((policy, metrics))

    feasible = [item for item in candidates if item[1]["constraints_met"]]
    if feasible:
        return min(
            feasible,
            key=lambda item: (
                item[1]["api_call_rate"],
                item[1]["harmful_wrong_queue_rate"],
                item[1]["review_rate"],
            ),
        )
    return min(
        candidates,
        key=lambda item: (
            item[1]["harmful_wrong_queue_rate"],
            max(0.0, item[1]["review_rate"] - maximum_review_rate),
            item[1]["api_call_rate"],
        ),
    )


def evaluate_policy(
    rows: Sequence[DatasetRow],
    local_intents: Sequence[str],
    local_confidences: Sequence[float],
    llm_routes: Mapping[str, LLMRoute],
    row_keys: Sequence[str],
    policy: HybridPolicy,
) -> tuple[dict[str, object], list[HybridDecision]]:
    return _evaluate_candidate(
        rows,
        local_intents,
        local_confidences,
        llm_routes,
        policy,
        row_keys,
    )
