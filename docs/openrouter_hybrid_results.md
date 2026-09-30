# OpenRouter GPT-6 Luna and Hybrid Results

## Execution record

The experiment was run on 30 September 2026 with the university-provided OpenRouter credit. The model slug was `openai/gpt-6-luna`. No synthetic queries were generated: the model only classified records from the cleaned BANKING77 subset.

The 60-row validation split was used to select thresholds. The 240-row test split was evaluated once after the policy was written to `artifacts/hybrid_policy.json`. Test results were not used to change prompts, thresholds, features, labels, or the queue taxonomy.

## Validation and policy selection

GPT-6 Luna alone made no harmful automatic queue errors on validation, but reviewed 22 of 60 records (36.7%), exceeding the 10% review budget. The validation search therefore selected a hybrid policy:

- Route locally when TF-IDF intent confidence is at least `0.15785826849376272`.
- Below that threshold, call GPT-6 Luna through OpenRouter.
- Review the query when the local and Luna queue predictions disagree, when Luna abstains, or when Luna is below its selected confidence threshold.

The frozen policy achieved a 5.0% harmful wrong-queue rate and 6.7% review rate on validation. It called Luna for 4 of 60 records.

## Held-out test result

| System | Harmful wrong-queue rate | Review rate | Automatic coverage | Intent accuracy | OpenRouter call rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority queue | 66.7% | 0.0% | 100.0% | N/A | 0.0% |
| Keyword rules | 25.8% | 0.0% | 100.0% | N/A | 0.0% |
| TF-IDF logistic regression | 5.0% | 0.0% | 100.0% | 89.2% | 0.0% |
| Frozen hybrid | **4.17%** | **3.33%** | **96.67%** | **89.17%** | **3.75%** |

The frozen hybrid met both project targets: harmful wrong-queue assignments were at or below 5% and human review was at or below 10%. Of 240 test queries, 8 were sent to review, 10 were harmful automatic wrong-queue assignments, and 232 were handled automatically. Selective queue accuracy among automatic decisions was 95.69%.

The hybrid called Luna for 9 of 240 test records. Those calls used 4,070 input tokens and 642 output tokens, with an estimated cost of $0.000728 at the dated OpenRouter rates. The complete 60-row validation run cost an estimated $0.004319. A separate one-query compatibility check preceded the recorded experiment and cost approximately $0.000067.

## Interpretation

The hybrid improves the safety metric modestly relative to the local TF-IDF model while using the rented model on fewer than 4% of test queries. The trade-off is operational: eight queries require human handling and ten automatic decisions still reach the wrong service queue. The prototype therefore reduces, but does not eliminate, routing risk.

The result does not establish production readiness. BANKING77 is public research data, the project uses only 12 of 77 intents, and the three queues are a project-specific aggregation. Live deployment would require bank-specific validation, monitoring for changing query language, privacy review, and a clear service-level agreement for the review queue.

## Provider documentation

- https://openrouter.ai/docs/quickstart
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://openrouter.ai/openai/gpt-6-luna
