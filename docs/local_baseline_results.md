# Local Baseline Results

## Evaluation status

The local baseline design, 12-intent taxonomy, queue mapping, sample counts, random seed, and TF-IDF logistic-regression configuration were fixed before the test run. The test results below are now evidence, not tuning feedback. Later foundation-model and hybrid implementations must follow the already documented design and use the validation split for thresholds or fallback selection.

## Results

| Split | System | Queue accuracy | Harmful wrong-queue rate | Intent accuracy | Review rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation, 60 queries | Majority queue | 33.3% | 66.7% | Not applicable | 0.0% |
| Validation, 60 queries | Keyword rules | 70.0% | 30.0% | Not applicable | 0.0% |
| Validation, 60 queries | TF-IDF logistic regression | 88.3% | 11.7% | 83.3% | 0.0% |
| Test, 240 queries | Majority queue | 33.3% | 66.7% | Not applicable | 0.0% |
| Test, 240 queries | Keyword rules | 74.2% | 25.8% | Not applicable | 0.0% |
| Test, 240 queries | TF-IDF logistic regression | 95.0% | 5.0% | 89.2% | 0.0% |

The test split contains 80 queries per queue. TF-IDF logistic regression made 12 wrong-queue assignments, exactly meeting the project's maximum 5% harmful wrong-queue target without human review. This result is a benchmark for the later foundation-model and hybrid systems; it does not establish that the target will hold on live banking traffic.

## What the baselines show

The balanced majority baseline achieves 33.3% queue accuracy. Keyword rules improve this to 74.2%, but default heavily to disputes and corrections when no configured phrase matches. They misroute 38 of 80 failed or pending transaction queries and 23 of 80 fees and pricing queries.

TF-IDF logistic regression reaches 95.0% queue accuracy and 89.2% fine-grained intent accuracy. The gap between these measures is expected because some wrong intent predictions remain inside the correct service queue. This is useful for the business task: the system can route a query correctly even when it confuses two specialist labels handled by the same team.

## Main TF-IDF errors

The largest fine-grained confusion is `transfer_fee_charged` predicted as `failed_transfer` in four test cases. This confusion crosses the fees and pricing and failed and pending queues, so it affects the primary metric. Other harmful errors include:

- "Is there a transfer fee?" predicted as `failed_transfer`.
- "I've been charged more than once for the same transaction" predicted as `transfer_fee_charged`.
- "It looks like someone else withdrew cash from my account, can you help?" predicted as `cash_withdrawal_charge`.
- "Somebody used my card to make a purchase" predicted as `declined_card_payment`.

These examples show why surface words such as "transfer," "charged," "cash," and "card" are insufficient by themselves. The later foundation-model comparison will test whether contextual interpretation reduces these cross-queue errors without exceeding the cost and review constraints.

## Reproduction

Run the following commands from the repository root:

```powershell
python scripts/prepare_banking77.py
banking-router evaluate-local
python -m unittest discover -s tests -v
```

The generated `manifest.json` records the source and split SHA-256 hashes. Detailed predictions and metrics are written to the ignored `results` directory so each evaluator can reproduce them locally.
