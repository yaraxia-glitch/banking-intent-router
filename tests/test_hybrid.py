"""Unit tests for threshold selection and hybrid routing decisions."""

import unittest

from banking_router.data import DatasetRow
from banking_router.hybrid import HybridPolicy, apply_policy, select_policy
from banking_router.llm import LLMRoute


class HybridPolicyTests(unittest.TestCase):
    def test_high_confidence_local_prediction_avoids_api(self):
        decision = apply_policy(
            local_intent="pending_transfer",
            local_confidence=0.9,
            llm_route=None,
            policy=HybridPolicy(0.7, 0.8),
        )
        self.assertEqual("local", decision.source)
        self.assertFalse(decision.reviewed)

    def test_cross_queue_disagreement_is_reviewed(self):
        decision = apply_policy(
            local_intent="pending_transfer",
            local_confidence=0.4,
            llm_route=LLMRoute(
                decision="route",
                intent="cash_withdrawal_charge",
                confidence=0.95,
                reason="Withdrawal fee.",
            ),
            policy=HybridPolicy(0.7, 0.8),
        )
        self.assertTrue(decision.reviewed)
        self.assertEqual("review_queue_disagreement", decision.source)

    def test_selection_uses_constraints(self):
        rows = [
            DatasetRow("a", "pending_transfer", "failed_and_pending_transactions", "x"),
            DatasetRow("b", "cash_withdrawal_charge", "fees_and_pricing", "x"),
        ]
        local_intents = ["pending_transfer", "pending_transfer"]
        local_confidences = [0.9, 0.1]
        llm_routes = {
            "a": LLMRoute("route", "pending_transfer", 0.9, "clear"),
            "b": LLMRoute("route", "cash_withdrawal_charge", 0.9, "clear"),
        }
        policy, metrics = select_policy(
            rows,
            local_intents,
            local_confidences,
            llm_routes,
            ["a", "b"],
            target_wrong_rate=0.0,
            maximum_review_rate=0.5,
        )
        self.assertTrue(metrics["constraints_met"])
        self.assertEqual(0.0, metrics["harmful_wrong_queue_rate"])
        self.assertGreater(policy.local_confidence_threshold, 0.1)


if __name__ == "__main__":
    unittest.main()
