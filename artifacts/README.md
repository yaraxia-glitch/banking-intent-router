# Model and Policy Artifacts

This directory contains the two frozen artifacts used by the final evaluation.

| File | Purpose |
| --- | --- |
| `tfidf_logistic.joblib` | Fitted local TF-IDF logistic-regression intent classifier trained on `data/processed/train.csv` |
| `hybrid_policy.json` | Validation-selected confidence thresholds, review rule, targets, and validation selection metrics |

The local model can be loaded without an API key. `hybrid_policy.json` records `selected_on: validation`; the final test command refuses to run unless a validation-selected policy exists.

Do not load an untrusted Joblib file. This artifact was produced by this repository from the checked-in public dataset.
