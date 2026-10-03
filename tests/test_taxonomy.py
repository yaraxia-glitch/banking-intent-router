"""Unit tests for the version-controlled intent-to-queue taxonomy."""

import unittest

from banking_router.taxonomy import INTENT_TO_QUEUE, QUEUE_TO_INTENTS, validate_taxonomy


class TaxonomyTests(unittest.TestCase):
    def test_taxonomy_has_three_balanced_queues(self):
        validate_taxonomy()
        self.assertEqual(3, len(QUEUE_TO_INTENTS))
        self.assertEqual({4}, {len(intents) for intents in QUEUE_TO_INTENTS.values()})
        self.assertEqual(12, len(INTENT_TO_QUEUE))

    def test_each_intent_appears_once(self):
        flattened = [intent for intents in QUEUE_TO_INTENTS.values() for intent in intents]
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main()
