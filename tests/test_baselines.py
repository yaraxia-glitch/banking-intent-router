import unittest

from banking_router.baselines import KeywordRuleRouter, MajorityQueueRouter, TfidfIntentRouter


class BaselineTests(unittest.TestCase):
    def test_majority_router_breaks_ties_deterministically(self):
        model = MajorityQueueRouter().fit(["queue_b", "queue_a"])
        self.assertEqual(["queue_a"], model.predict(["anything"]))

    def test_keyword_router_matches_each_queue(self):
        queues = [
            "disputes_and_corrections",
            "failed_and_pending_transactions",
            "fees_and_pricing",
        ]
        model = KeywordRuleRouter().fit(queues)
        texts = [
            "I did not make this cash withdrawal",
            "My card payment is still pending",
            "Why was I charged a transfer fee",
        ]
        self.assertEqual(
            [
                "disputes_and_corrections",
                "failed_and_pending_transactions",
                "fees_and_pricing",
            ],
            model.predict(texts),
        )

    def test_tfidf_router_learns_small_fixture(self):
        texts = [
            "card fee charged",
            "transfer fee cost",
            "cash withdrawal not recognised",
            "card payment not mine",
            "transfer is pending",
            "card payment declined",
        ]
        intents = [
            "card_payment_fee_charged",
            "transfer_fee_charged",
            "cash_withdrawal_not_recognised",
            "card_payment_not_recognised",
            "pending_transfer",
            "declined_card_payment",
        ]
        model = TfidfIntentRouter(seed=6201).fit(texts, intents)
        predicted = model.predict_intents(texts)
        self.assertEqual(intents, predicted)


if __name__ == "__main__":
    unittest.main()

