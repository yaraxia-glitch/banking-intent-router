"""Unit tests for routing metrics and review-aware evaluation."""

import unittest

from banking_router.evaluation import calculate_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_apply_review_to_wrong_queue_rate(self):
        metrics = calculate_metrics(
            true_queues=["a", "a", "b", "b"],
            predicted_queues=["a", "b", "a", "b"],
            reviewed=[False, True, False, False],
        )
        self.assertEqual(1, metrics["review_count"])
        self.assertAlmostEqual(0.25, metrics["review_rate"])
        self.assertAlmostEqual(0.25, metrics["harmful_wrong_queue_rate"])
        self.assertAlmostEqual(2 / 3, metrics["selective_queue_accuracy"])

    def test_metrics_reject_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            calculate_metrics(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
