"""GPT-6 Luna routing through OpenRouter with strict structured output."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Final

from banking_router.taxonomy import SELECTED_INTENTS, queue_for_intent


DEFAULT_MODEL: Final[str] = "openai/gpt-6-luna"
OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
UNKNOWN_INTENT: Final[str] = "unknown"

INTENT_DESCRIPTIONS: Final[dict[str, str]] = {
    "card_payment_not_recognised": "The customer does not recognise a card purchase.",
    "cash_withdrawal_not_recognised": "The customer does not recognise an ATM cash withdrawal.",
    "direct_debit_payment_not_recognised": "The customer does not recognise a direct debit.",
    "transaction_charged_twice": "The same transaction or card payment appears twice.",
    "declined_card_payment": "A card payment was declined.",
    "pending_card_payment": "A card payment is still pending.",
    "failed_transfer": "A bank transfer failed or did not go through.",
    "pending_transfer": "A bank transfer is still pending.",
    "card_payment_fee_charged": "A fee was charged for a card payment.",
    "cash_withdrawal_charge": "A fee was charged for an ATM cash withdrawal.",
    "exchange_charge": "A currency exchange or conversion fee was charged.",
    "transfer_fee_charged": "A fee was charged for a bank transfer.",
}


@dataclass(frozen=True)
class LLMRoute:
    decision: str
    intent: str
    confidence: float
    reason: str
    model: str = DEFAULT_MODEL
    response_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def reviewed(self) -> bool:
        return self.decision == "review" or self.intent == UNKNOWN_INTENT

    @property
    def queue(self) -> str:
        if self.intent == UNKNOWN_INTENT:
            return "REVIEW"
        return queue_for_intent(self.intent)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMRoute":
        return cls(
            decision=str(data["decision"]),
            intent=str(data["intent"]),
            confidence=float(data["confidence"]),
            reason=str(data.get("reason", "")),
            model=str(data.get("model", DEFAULT_MODEL)),
            response_id=str(data.get("response_id", "")),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
        )


def _response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["route", "review"]},
            "intent": {
                "type": "string",
                "enum": [*SELECTED_INTENTS, UNKNOWN_INTENT],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["decision", "intent", "confidence", "reason"],
        "additionalProperties": False,
    }


def system_prompt() -> str:
    definitions = "\n".join(
        f"- {intent}: {INTENT_DESCRIPTIONS[intent]}" for intent in SELECTED_INTENTS
    )
    return f"""You route short banking support queries. Choose only from the fixed taxonomy below.

{definitions}

Return decision='route' only when one intent is clearly supported by the text. Return
decision='review' and intent='unknown' when the query is outside this taxonomy, lacks
enough detail, or is genuinely ambiguous across queues. Do not infer fraud, identity,
account status, or facts that the customer did not state. Confidence is a number from
0 to 1 representing confidence in the selected intent. Keep the reason short and do
not answer the customer."""


class OpenRouterLunaRouter:
    """Thin adapter around OpenRouter's OpenAI-compatible API.

    A client can be injected for unit tests. ``from_environment`` prefers
    ``OPENROUTER_API_KEY`` and falls back to ``OPENAI_API_KEY`` so an already-set
    university-provided key continues to work without being written to disk.
    """

    def __init__(self, client: Any, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_environment(cls, model: str = DEFAULT_MODEL) -> "OpenRouterLunaRouter":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the API extra first: python -m pip install -e '.[openai]'"
            ) from exc
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY before running"
            )
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers={"X-OpenRouter-Title": "PE6201 Banking Intent Router"},
        )
        return cls(client, model=model)

    def route(self, text: str) -> LLMRoute:
        if not text.strip():
            raise ValueError("A non-empty query is required")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "banking_route",
                    "strict": True,
                    "schema": _response_schema(),
                }
            },
            max_tokens=200,
            extra_body={
                "reasoning": {"effort": "low"},
                "provider": {"require_parameters": True},
            },
        )
        try:
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError("The model refused the routing request")
            payload = json.loads(message.content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenRouter did not return a valid structured route") from exc

        decision = str(payload.get("decision", ""))
        intent = str(payload.get("intent", ""))
        confidence = float(payload.get("confidence", -1))
        reason = str(payload.get("reason", ""))
        if decision not in {"route", "review"}:
            raise RuntimeError(f"Unexpected decision: {decision!r}")
        if intent not in {*SELECTED_INTENTS, UNKNOWN_INTENT}:
            raise RuntimeError(f"Unexpected intent: {intent!r}")
        if not 0 <= confidence <= 1:
            raise RuntimeError(f"Confidence is outside [0, 1]: {confidence}")
        if decision == "route" and intent == UNKNOWN_INTENT:
            raise RuntimeError("A route decision cannot use the unknown intent")

        usage = getattr(response, "usage", None)
        return LLMRoute(
            decision=decision,
            intent=intent,
            confidence=confidence,
            reason=reason,
            model=self.model,
            response_id=str(getattr(response, "id", "")),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
