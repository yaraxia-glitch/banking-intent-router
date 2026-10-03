# Risk Aware Banking Query Routing

This repository evaluates small, reproducible systems that route short online-banking support queries to one of three service queues. It compares a majority-queue sanity check, a deterministic keyword baseline, a TF-IDF logistic-regression classifier, GPT-6 Luna, and a review-budgeted hybrid policy.

## Project question

Can a hybrid router keep harmful wrong-queue assignments at or below 5% of all test queries while sending no more than 10% of queries to human review?

The prototype does not answer customers, execute transactions, infer fraud, or process live personal data.

## Quick verification with no API call

The repository includes the public data snapshot, trained local artifact, cached model classifications, and final evaluation outputs used in the report. A marker or reviewer can verify the complete project without an OpenRouter key:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\verify_project.py
```

The expected final lines are `Ran 19 tests`, `OK`, and `VERIFICATION PASSED`. This command is read-only and does not spend API credit.

For a concise recorded-demo view of the final metrics and one reviewed example, run:

```powershell
.\.venv\Scripts\python.exe scripts\demo_summary.py
```

This second command also reads only checked-in files and makes no network or API call.

## Dataset

The project uses the public [BANKING77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), which contains 13,083 English customer-service queries across 77 intents. The source repository is licensed under CC BY 4.0.

The preparation command selects 12 intents and produces a deterministic 540-query subset with seed `6201`:

- 240 training queries: 20 per intent from the official training split
- 60 validation queries: 5 per intent from the official training split
- 240 test queries: 20 per intent from the official test split

The official test examples remain separate from threshold selection and model selection.

## Queue taxonomy

| Queue | BANKING77 intents |
| --- | --- |
| Disputes and corrections | `card_payment_not_recognised`, `cash_withdrawal_not_recognised`, `direct_debit_payment_not_recognised`, `transaction_charged_twice` |
| Failed and pending transactions | `declined_card_payment`, `pending_card_payment`, `failed_transfer`, `pending_transfer` |
| Fees and pricing | `card_payment_fee_charged`, `cash_withdrawal_charge`, `exchange_charge`, `transfer_fee_charged` |

The single source of truth for this mapping is `src/banking_router/taxonomy.py`.

## Product documentation

The complete product view is in [`docs/product_documentation.md`](docs/product_documentation.md). It records the persona, input and output contract, system architecture, owned and rented components, target metrics, measured results, scope, and operating controls.

Supporting repository guides:

- [`data/README.md`](data/README.md) explains the source, licence, cleaning lineage, schemas, split sizes, and data files.
- [`results/README.md`](results/README.md) explains the evaluation protocol, metric definitions, and every checked-in result file.
- [`artifacts/README.md`](artifacts/README.md) explains the trained local model and frozen hybrid policy.
- [`docs/code_map.md`](docs/code_map.md) maps the commands and Python modules to their responsibilities.

## Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional OpenAI SDK only for the rented-model experiment. The SDK is configured to use OpenRouter's OpenAI-compatible endpoint:

```powershell
python -m pip install -e ".[openai]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Run the local experiment

Download the untouched official files, run the cleaning audit, and create the fixed subset:

```powershell
python scripts/prepare_banking77.py
```

The same workflow can be run as three inspectable stages:

```powershell
banking-router download-data
banking-router clean-data
banking-router prepare-data
```

Train and evaluate all local baselines:

```powershell
banking-router evaluate-local
```

This writes:

- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/manifest.json`
- `data/cleaned/banking77_train_cleaned.csv`
- `data/cleaned/banking77_test_cleaned.csv`
- `data/cleaned/quarantine.csv`
- `data/cleaned/cleaning_report.json`
- `docs/data_cleaning_report.md`
- `artifacts/tfidf_logistic.joblib`
- `results/local_metrics.json`
- `results/local_predictions.csv`

## Local baseline results

The protocol and model settings were fixed before the test run. The 240-query test split is balanced across the three queues and is not used for further feature, keyword, or hyperparameter changes.

| System | Validation queue accuracy | Test queue accuracy | Test harmful wrong-queue rate | Test intent accuracy |
| --- | ---: | ---: | ---: | ---: |
| Majority queue | 33.3% | 33.3% | 66.7% | Not applicable |
| Keyword rules | 70.0% | 74.2% | 25.8% | Not applicable |
| TF-IDF logistic regression | 88.3% | 95.0% | 5.0% | 89.2% |

All three local systems route every query and therefore have a 0% review rate at this stage. The detailed interpretation and main error examples are recorded in `docs/local_baseline_results.md`.

## Final hybrid result

The GPT-6 Luna experiment was run through OpenRouter using the university-provided credit. Thresholds were selected on validation and frozen before the final test run.

| System | Test harmful wrong-queue rate | Test review rate | Automatic coverage | OpenRouter call rate |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF logistic regression | 5.0% | 0.0% | 100.0% | 0.0% |
| Frozen TF-IDF + GPT-6 Luna hybrid | **4.17%** | **3.33%** | **96.67%** | **3.75%** |

The hybrid met the project constraints of no more than 5% harmful wrong-queue assignments and no more than 10% human review. The full protocol, validation result, held-out test result, cost, and limitations are documented in `docs/openrouter_hybrid_results.md`.

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Run the read-only end-to-end verification after the experiment outputs exist:

```powershell
python scripts/verify_project.py
```

This checks source and generated-file hashes, row counts, split isolation, API-cache counts, the frozen validation policy, final constraints, absence of embedded API keys, and the complete automated test suite. It does not make API calls or spend credit.

## Run the rented-model and hybrid experiment

Set the university-provided OpenRouter key as `OPENROUTER_API_KEY` in your local environment; do not put it in a source file or commit it. For compatibility, the code also accepts an existing `OPENAI_API_KEY` variable and still sends it only to OpenRouter. Then run the stages in order:

```powershell
banking-router evaluate-luna-validation
banking-router select-hybrid-policy
banking-router evaluate-final
```

The first command sends only the 60 validation queries to GPT-6 Luna. The second command selects thresholds using validation only and freezes them in `artifacts/hybrid_policy.json`. The final command evaluates that frozen policy on the 240-query test subset and calls Luna only for low-confidence local predictions. API responses are cached incrementally in JSONL files, so an interrupted run can resume. The assessed cache files are checked in as evaluation evidence; they contain only public BANKING77 queries and model classifications, never the API key.

No synthetic examples are used at any stage. GPT-6 Luna is accessed through OpenRouter and only classifies cleaned queries originating in the official BANKING77 files. The model and policy design is documented in `docs/model_design.md`.

## Reproducibility controls

- Dataset sampling uses seed `6201` and sorted source rows.
- Original source files remain unchanged and are hashed with SHA-256.
- Cleaning applies Unicode NFKC and whitespace normalisation, label validation, duplicate removal, train-test leakage prevention, and a documented PII-pattern scan.
- Every removed row is retained in `quarantine.csv` with its source row number and reason.
- The source file hashes and generated split hashes are stored in `manifest.json`.
- The intent-to-queue map is version controlled and unit tested.
- The same processed test file is used for every model.
- The exact public dataset snapshot, cleaned and processed data, trained local artifact, cached classifications, and evaluation outputs used for the submission are version controlled.
- API secrets, `.env` files, virtual environments, and Python caches are excluded from Git.

## Current limitations

- BANKING77 is a research dataset, not live operational bank traffic.
- The three queues are a project-specific aggregation of 12 fine-grained intents.
- The keyword baseline is deliberately simple and is not a production rule engine.
- The measured OpenRouter and hybrid results are specific to the dated model, provider routing, fixed subset, and frozen policy; they should not be generalized to live bank traffic.

## Citation

Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. (2020). Efficient Intent Detection with Dual Sentence Encoders. Proceedings of the 2nd Workshop on NLP for Conversational AI. [Paper](https://arxiv.org/abs/2003.04807).
