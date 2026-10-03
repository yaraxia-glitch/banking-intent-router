# Data and Cleaning Evidence

This directory checks in the exact public BANKING77 files and deterministic derivatives used in the assessed experiment. No synthetic examples are present.

## Source and licence

- Dataset: BANKING77
- Original repository: https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data
- Licence recorded by the source repository: CC BY 4.0
- Raw rows: 13,083 across 77 intents
- Sampling seed: `6201`

`processed/manifest.json` records the source URL, licence, cleaning policy, split counts, and SHA-256 hashes of the raw and generated data files.

## raw

Untouched files downloaded from the official PolyAI-LDN repository:

- `banking77_train.csv`
- `banking77_test.csv`
- `banking77_categories.json`

The preparation script never edits these files. Their SHA-256 hashes are stored in `data/processed/manifest.json`.

## cleaned

Full cleaned versions of the original train and test files, plus the audit trail:

- `banking77_train_cleaned.csv`
- `banking77_test_cleaned.csv`
- `quarantine.csv`
- `cleaning_report.json`

The cleaning pipeline retained 13,071 rows and quarantined 12 rows: five within-split duplicates and seven training rows that overlapped the official test split. It normalised 570 texts and detected no configured PII pattern matches.

## processed

The fixed 12-intent experiment subset:

- `train.csv`
- `validation.csv`
- `test.csv`
- `manifest.json`

The fixed experiment subset contains 12 intents mapped to three queues:

| Split | Rows | Purpose |
| --- | ---: | --- |
| Train | 240 | Fit the local TF-IDF logistic-regression model |
| Validation | 60 | Select thresholds and freeze the hybrid policy |
| Test | 240 | One held-out final evaluation after policy selection |

The three CSV files use the fields `text`, `intent`, `queue`, and `source_split`. The raw CSV files use `text` and `category`; the cleaned files additionally record the source row, text hash, and PII flags. Every selected intent has the same number of rows within each split, and texts do not overlap across train, validation, and test.

## Reproduce the data

Run `python scripts/prepare_banking77.py` from the repository root to reproduce all generated data.

The command downloads the original files only from the recorded PolyAI source, cleans them, writes the quarantine audit, and recreates the fixed splits. Running it does not call OpenRouter.
