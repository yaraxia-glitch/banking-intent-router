# Code Map

This map explains the repository at file and module level so a reviewer can scan the implementation without reading every function.

## Entry points

| Path | Responsibility |
| --- | --- |
| `scripts/prepare_banking77.py` | Run download, cleaning, audit, and deterministic sampling in one command |
| `scripts/verify_project.py` | Perform the read-only end-to-end evidence and test verification used in the demo |
| `scripts/demo_summary.py` | Print a concise offline summary of final metrics and one reviewed example for the recording |
| `src/banking_router/__main__.py` | Start the `banking-router` command-line interface |
| `src/banking_router/cli.py` | Define the data, local evaluation, Luna validation, policy selection, and final evaluation commands |

## Implementation modules

| Module | Responsibility |
| --- | --- |
| `taxonomy.py` | Version-controlled mapping from 12 BANKING77 intents to three service queues |
| `cleaning.py` | Unicode and whitespace normalisation, category validation, duplicate handling, PII-pattern scan, leakage quarantine, and cleaning report |
| `data.py` | Official-source download, deterministic balanced sampling, split writing, and manifest generation |
| `baselines.py` | Majority, keyword, and TF-IDF logistic-regression routing models |
| `evaluation.py` | Shared review-aware metrics and local baseline evaluation |
| `llm.py` | Strict structured-output adapter for GPT-6 Luna through OpenRouter |
| `hybrid.py` | Validation-only threshold search, frozen policy representation, and final route/review logic |
| `experiments.py` | Cached Luna validation and held-out hybrid test orchestration |

## Tests

The seven test modules cover the taxonomy, cleaning rules, deterministic splits, baseline behaviour, metric calculations, structured LLM parsing, and hybrid-policy constraints. Run all 19 tests with:

```powershell
python -m unittest discover -s tests -v
```

`scripts/verify_project.py` runs the same suite after checking data and experiment evidence.
