# Evaluation Evidence

This directory contains the exact evaluation outputs used in the report and recorded demo. All query text originates in the public BANKING77 dataset. No live customer data or API key is stored here.

## Evaluation protocol

1. Train the local TF-IDF logistic-regression model on 240 training rows.
2. Evaluate the majority, keyword, and TF-IDF baselines on the fixed validation and test splits.
3. Classify all 60 validation rows with GPT-6 Luna through OpenRouter.
4. Select confidence thresholds on validation only, subject to at most 5% harmful wrong-queue assignments and at most 10% review.
5. Freeze the policy in `artifacts/hybrid_policy.json`.
6. Evaluate the frozen policy once on the 240 held-out test rows. Luna is called only when the local confidence is below the frozen threshold.

## Files

| File | Contents |
| --- | --- |
| `local_metrics.json` | Validation and test metrics for the majority, keyword, and TF-IDF baselines |
| `local_predictions.csv` | Row-level local predictions and confidence values |
| `luna_validation.jsonl` | Cached structured Luna classifications for all 60 validation rows |
| `luna_validation_metrics.json` | Luna-only validation metrics, token counts, and dated cost estimate |
| `luna_test.jsonl` | Nine cached Luna classifications requested by the frozen test policy |
| `final_predictions.csv` | Row-level final decisions, true labels, review flags, decision source, and confidence values |
| `final_metrics.json` | Final held-out hybrid metrics, confusion counts, coverage, API usage, and dated cost estimate |

## Metric definitions

- **Harmful wrong-queue rate**: automatic decisions assigned to the wrong queue divided by all test rows. A reviewed row is not counted as a harmful automatic assignment.
- **Review rate**: rows returned as `REVIEW` divided by all rows.
- **Automatic coverage**: rows assigned automatically divided by all rows.
- **Selective queue accuracy**: correct queue assignments divided by automatically assigned rows, excluding reviewed rows.
- **Intent accuracy**: predicted fine-grained intent divided by all rows.

## Final held-out result

| Metric | Target | Reached |
| --- | ---: | ---: |
| Harmful wrong-queue rate | At most 5% | 4.17% |
| Human-review rate | At most 10% | 3.33% |
| Automatic coverage | Reported, no minimum | 96.67% |
| Selective queue accuracy | Reported, no minimum | 95.69% |
| OpenRouter call rate | Minimise after safety constraints | 3.75% (9 of 240) |

The result supports a prototype decision only. The validation set has 60 rows, the test set has 240 rows, and the balanced research subset does not represent live bank traffic.

## Verify without spending API credit

From the repository root, run:

```powershell
python scripts/verify_project.py
```

The command validates hashes, row counts, split isolation, cache counts, the frozen policy, final constraints, secret hygiene, and all automated tests. It reads the checked-in evidence and makes no network or OpenRouter call.

For a short human-readable view suitable for the recorded demo, run:

```powershell
python scripts/demo_summary.py
```

It prints the target and reached metrics plus one reviewed public-dataset example. It also makes no network or API call.
