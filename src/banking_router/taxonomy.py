"""Version-controlled intent and queue taxonomy."""

from __future__ import annotations

from collections import Counter
from typing import Final


QUEUE_TO_INTENTS: Final[dict[str, tuple[str, ...]]] = {
    "disputes_and_corrections": (
        "card_payment_not_recognised",
        "cash_withdrawal_not_recognised",
        "direct_debit_payment_not_recognised",
        "transaction_charged_twice",
    ),
    "failed_and_pending_transactions": (
        "declined_card_payment",
        "pending_card_payment",
        "failed_transfer",
        "pending_transfer",
    ),
    "fees_and_pricing": (
        "card_payment_fee_charged",
        "cash_withdrawal_charge",
        "exchange_charge",
        "transfer_fee_charged",
    ),
}

INTENT_TO_QUEUE: Final[dict[str, str]] = {
    intent: queue
    for queue, intents in QUEUE_TO_INTENTS.items()
    for intent in intents
}

SELECTED_INTENTS: Final[tuple[str, ...]] = tuple(INTENT_TO_QUEUE)
QUEUES: Final[tuple[str, ...]] = tuple(QUEUE_TO_INTENTS)


def queue_for_intent(intent: str) -> str:
    """Return the service queue for an allowed intent."""

    try:
        return INTENT_TO_QUEUE[intent]
    except KeyError as exc:
        raise ValueError(f"Unsupported intent: {intent}") from exc


def validate_taxonomy() -> None:
    """Raise ValueError if the project taxonomy violates its fixed design."""

    if len(SELECTED_INTENTS) != 12:
        raise ValueError("The project taxonomy must contain exactly 12 intents")
    if len(QUEUES) != 3:
        raise ValueError("The project taxonomy must contain exactly three queues")
    counts = Counter(INTENT_TO_QUEUE.values())
    if set(counts.values()) != {4}:
        raise ValueError("Each queue must contain exactly four intents")

