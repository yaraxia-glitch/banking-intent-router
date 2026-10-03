"""Unit tests for deterministic sampling and split isolation."""

import unittest
from collections import Counter

from banking_router.data import DatasetRow, build_splits
from banking_router.taxonomy import SELECTED_INTENTS, queue_for_intent


def make_rows(per_intent, source_split):
    return [
        DatasetRow(
            text=f"{intent} example {index:03d}",
            intent=intent,
            queue=queue_for_intent(intent),
            source_split=source_split,
        )
        for intent in SELECTED_INTENTS
        for index in range(per_intent)
    ]


class DataTests(unittest.TestCase):
    def test_build_splits_has_fixed_balanced_counts(self):
        splits = build_splits(
            make_rows(40, "official_train"),
            make_rows(30, "official_test"),
            seed=6201,
        )
        self.assertEqual(240, len(splits["train"]))
        self.assertEqual(60, len(splits["validation"]))
        self.assertEqual(240, len(splits["test"]))
        self.assertEqual({20}, set(Counter(row.intent for row in splits["train"]).values()))
        self.assertEqual({5}, set(Counter(row.intent for row in splits["validation"]).values()))
        self.assertEqual({20}, set(Counter(row.intent for row in splits["test"]).values()))

    def test_build_splits_is_deterministic(self):
        train = make_rows(40, "official_train")
        test = make_rows(30, "official_test")
        first = build_splits(train, test, seed=6201)
        second = build_splits(train, test, seed=6201)
        self.assertEqual(first, second)

    def test_training_and_validation_do_not_overlap(self):
        splits = build_splits(
            make_rows(40, "official_train"),
            make_rows(30, "official_test"),
            seed=6201,
        )
        train_text = {row.text for row in splits["train"]}
        validation_text = {row.text for row in splits["validation"]}
        self.assertTrue(train_text.isdisjoint(validation_text))


if __name__ == "__main__":
    unittest.main()
