# GPT-6 Luna and Hybrid Model Design

## Purpose

The rented model classifies the same cleaned, human-authored BANKING77 queries used by the local baselines. It does not generate training or test data. The prompt contains only the fixed intent definitions and the query being classified; it contains no synthetic examples.

## Rented model

The implementation uses `openai/gpt-6-luna` through the university-provided OpenRouter credit and OpenRouter's OpenAI-compatible Chat Completions endpoint. It requests low reasoning effort and a strict JSON schema. The model must return a route/review decision, one of the 12 fixed intents or `unknown`, a confidence score, and a short reason. The service queue is derived in local code from the version-controlled taxonomy rather than trusted as a free-form model output.

OpenRouter lists GPT-6 Luna as suitable for high-volume classification and shows support for structured outputs. Pricing checked on 30 September 2026 was $0.10 per million input tokens and $0.50 per million output tokens. Pricing is dated in the result files because it can change. The API request sets `provider.require_parameters=true` so OpenRouter routes only to endpoints that support the required JSON schema.

Official sources:

- https://openrouter.ai/docs/quickstart
- https://openrouter.ai/docs/guides/features/structured-outputs
- https://openrouter.ai/openai/gpt-6-luna

## Hybrid policy and test discipline

1. Train the TF-IDF logistic-regression model on the 240-row training subset.
2. Run Luna on all 60 validation rows and cache each response immediately.
3. Search local-confidence and Luna-confidence thresholds using validation only.
4. Retain only policies with harmful wrong-queue rate at or below 5% and review rate at or below 10%. Among feasible policies, choose the one with the lowest API-call rate, then the lowest harmful error and review rates.
5. Freeze the selected thresholds in `artifacts/hybrid_policy.json`.
6. Run the frozen policy once on the 240-row test subset. Luna is called only for rows below the frozen local-confidence threshold.

When the local and Luna queue predictions disagree below the local threshold, the query is sent to review. Luna can also abstain on an unclear or out-of-scope query. If no validation policy satisfies both constraints, the code records `constraints_met: false`; it does not claim success.

## Reproducibility and privacy controls

- API results are appended to JSONL after each call, so interrupted runs resume without paying for completed rows again.
- The university-provided key is read from `OPENROUTER_API_KEY`, with `OPENAI_API_KEY` accepted only as a compatibility fallback. The key is never stored in the repository.
- Generated result caches, trained artifacts, and secrets are excluded from Git.
- Token counts and a dated cost estimate are written to the metrics files.
- The final test command refuses to run until a validation-selected policy exists.
