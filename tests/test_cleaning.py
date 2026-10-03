"""Unit tests for text normalisation, PII handling, and cleaning audits."""

import unittest

from banking_router.cleaning import RawRow, clean_rows, normalise_text, redact_pii


CATEGORIES = [f"category_{index}" for index in range(77)]


class CleaningTests(unittest.TestCase):
    def test_normalise_text_collapses_unicode_and_whitespace(self):
        self.assertEqual("Payment pending", normalise_text("  Payment\u00a0\t pending  "))

    def test_redact_pii_replaces_detected_values(self):
        text, flags = redact_pii("Email me at person@example.com or use GB82WEST12345698765432")
        self.assertIn("email", flags)
        self.assertIn("iban", flags)
        self.assertNotIn("person@example.com", text)
        self.assertNotIn("GB82WEST12345698765432", text)

    def test_cleaning_quarantines_duplicates_and_training_leakage(self):
        train = [
            RawRow("Repeated text", CATEGORIES[0], "official_train", 2),
            RawRow(" repeated   text ", CATEGORIES[0], "official_train", 3),
            RawRow("Test overlap", CATEGORIES[1], "official_train", 4),
            RawRow("Train only", CATEGORIES[2], "official_train", 5),
        ]
        test = [
            RawRow("test overlap", CATEGORIES[1], "official_test", 2),
            RawRow("Test only", CATEGORIES[3], "official_test", 3),
        ]
        cleaned, quarantine, report = clean_rows(train, test, CATEGORIES)
        self.assertEqual(2, len(cleaned["train"]))
        self.assertEqual(2, len(cleaned["test"]))
        reasons = [row.reason for row in quarantine]
        self.assertEqual(1, reasons.count("duplicate_within_split"))
        self.assertEqual(1, reasons.count("cross_split_leakage_training_removed"))
        self.assertEqual(2, report["changes"]["quarantined_total"])

    def test_cleaning_quarantines_invalid_category(self):
        train = [RawRow("Example", "not_official", "official_train", 2)]
        cleaned, quarantine, _ = clean_rows(train, [], CATEGORIES)
        self.assertEqual([], cleaned["train"])
        self.assertEqual("invalid_category", quarantine[0].reason)


if __name__ == "__main__":
    unittest.main()
