"""Print a concise, offline summary of the checked-in final evaluation."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "final_metrics.json"
PREDICTIONS_PATH = ROOT / "results" / "final_predictions.csv"


def percent(value: float) -> str:
    """Format a decimal rate as a percentage with two decimal places."""

    return f"{100 * value:.2f}%"


def first_reviewed_prediction() -> dict[str, str]:
    """Return the first checked-in prediction that was sent to review."""

    with PREDICTIONS_PATH.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["reviewed"].lower() == "true":
                return row
    raise RuntimeError("No reviewed prediction was found")


def main() -> None:
    """Print final metrics and one review example without any network call."""

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = int(metrics["rows"])
    review_count = int(metrics["review_count"])
    automatic_count = rows - review_count
    correct_automatic = round(
        automatic_count * float(metrics["selective_queue_accuracy"])
    )
    example = first_reviewed_prediction()

    print("PE6201 FINAL DEMO SUMMARY")
    print("=" * 60)
    print(f"Held-out test rows:             {rows}")
    print(
        "Harmful wrong queue:           "
        f"{metrics['harmful_wrong_queue_count']}/{rows} = "
        f"{percent(float(metrics['harmful_wrong_queue_rate']))} (target <= 5%)"
    )
    print(
        "Human review:                  "
        f"{review_count}/{rows} = {percent(float(metrics['review_rate']))} "
        "(limit <= 10%)"
    )
    print(
        "Automatic coverage:            "
        f"{automatic_count}/{rows} = {percent(float(metrics['automatic_coverage']))}"
    )
    print(
        "Selective queue accuracy:      "
        f"{correct_automatic}/{automatic_count} = "
        f"{percent(float(metrics['selective_queue_accuracy']))}"
    )
    print(f"Intent accuracy:                {percent(float(metrics['intent_accuracy']))}")
    print(
        "OpenRouter calls:               "
        f"{metrics['api_call_count']}/{rows} = "
        f"{percent(float(metrics['api_call_rate']))}"
    )
    print(f"Estimated final-test API cost:  ${metrics['estimated_standard_cost_usd']:.6f}")
    print()
    print("ONE REVIEWED PUBLIC-DATASET EXAMPLE")
    print("=" * 60)
    print("Text:            " + textwrap.shorten(example["text"], width=120))
    print("True intent:     " + example["true_intent"])
    print("Predicted intent:" + " " + example["predicted_intent"])
    print("True queue:      " + example["true_queue"])
    print("Predicted queue: " + example["predicted_queue"])
    print("Decision source: " + example["source"])
    print()
    print("OFFLINE DEMO COMPLETE: no API call was made")


if __name__ == "__main__":
    main()
