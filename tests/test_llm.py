"""Unit tests for structured GPT-6 Luna routing through OpenRouter."""

import json
import unittest
from types import SimpleNamespace

from banking_router.llm import OpenRouterLunaRouter


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            id="resp_test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(self.payload),
                        refusal=None,
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=17),
        )


class FakeChat:
    def __init__(self, responses):
        self.completions = responses


class LunaRouterTests(unittest.TestCase):
    def test_structured_route_is_validated_and_mapped(self):
        responses = FakeResponses(
            {
                "decision": "route",
                "intent": "pending_transfer",
                "confidence": 0.92,
                "reason": "The transfer is explicitly pending.",
            }
        )
        router = OpenRouterLunaRouter(SimpleNamespace(chat=FakeChat(responses)))
        result = router.route("Why is my transfer still pending?")

        self.assertEqual("failed_and_pending_transactions", result.queue)
        self.assertFalse(result.reviewed)
        self.assertEqual(123, result.input_tokens)
        self.assertEqual("openai/gpt-6-luna", responses.last_request["model"])
        schema = responses.last_request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])

    def test_unknown_route_must_be_reviewed(self):
        responses = FakeResponses(
            {
                "decision": "route",
                "intent": "unknown",
                "confidence": 0.5,
                "reason": "Unclear.",
            }
        )
        router = OpenRouterLunaRouter(SimpleNamespace(chat=FakeChat(responses)))
        with self.assertRaises(RuntimeError):
            router.route("hello")


if __name__ == "__main__":
    unittest.main()
