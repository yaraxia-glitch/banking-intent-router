# BANKING77 Data Cleaning Report

## Provenance

The script downloads the original `train.csv`, `test.csv`, and `categories.json` files directly from the PolyAI-LDN task-specific-datasets repository. Raw files are retained without modification. SHA-256 hashes are recorded in the experiment manifest.

## Cleaning policy

- Apply Unicode NFKC normalisation.
- Replace control characters with spaces, collapse repeated whitespace, and trim edges.
- Validate every label against the official 77-category file.
- Quarantine empty rows and conflicting-label records.
- Keep the first normalised duplicate within a split and quarantine later duplicates.
- Preserve the official test row when the same normalised text appears in train and test; quarantine the training copy to prevent leakage.
- Scan for email addresses, URLs, IBAN-like values, and long digit sequences; replace detected values with typed placeholders.

## Audit results

| Measure | Count |
| --- | ---: |
| Original train rows | 10,003 |
| Original test rows | 3,080 |
| Cleaned train rows | 9,992 |
| Cleaned test rows | 3,079 |
| Text rows changed by normalisation | 570 |
| Total quarantined rows | 12 |
| Rows with PII-like values redacted | 0 |

### Quarantine reasons

- `cross_split_leakage_training_removed`: 7
- `duplicate_within_split`: 5

### PII scan

- No configured PII-like patterns were detected.

## Outputs

- `data/cleaned/banking77_train_cleaned.csv`
- `data/cleaned/banking77_test_cleaned.csv`
- `data/cleaned/quarantine.csv`
- `data/cleaned/cleaning_report.json`

The cleaning process does not prove that the source is free of personal data. It documents the patterns checked and preserves an audit trail for every row removed by the script.
