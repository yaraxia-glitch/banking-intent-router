# Data Directories

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

## processed

The fixed 12-intent experiment subset:

- `train.csv`
- `validation.csv`
- `test.csv`
- `manifest.json`

Run `python scripts/prepare_banking77.py` from the repository root to reproduce all generated data.

