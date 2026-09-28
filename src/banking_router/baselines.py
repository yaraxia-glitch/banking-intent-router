"""Deterministic and narrow machine-learning baselines."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

from banking_router.taxonomy import queue_for_intent


class MajorityQueueRouter:
    def __init__(self) -> None:
        self.queue_: str | None = None

    def fit(self, queues: Sequence[str]) -> "MajorityQueueRouter":
        if not queues:
            raise ValueError("At least one training queue is required")
        counts = Counter(queues)
        self.queue_ = sorted(counts, key=lambda queue: (-counts[queue], queue))[0]
        return self

    def predict(self, texts: Sequence[str]) -> list[str]:
        if self.queue_ is None:
            raise RuntimeError("MajorityQueueRouter must be fitted before prediction")
        return [self.queue_] * len(texts)


class KeywordRuleRouter:
    """A deliberately small, transparent non-AI queue router."""

    KEYWORDS = {
        "disputes_and_corrections": (
            "not recognised",
            "not recognized",
            "do not recognise",
            "don't recognise",
            "did not make",
            "wasn't me",
            "not mine",
            "charged twice",
            "charge twice",
            "duplicate",
            "duplicated",
            "twice",
        ),
        "failed_and_pending_transactions": (
            "pending",
            "declined",
            "failed",
            "not gone through",
            "did not go through",
            "hasn't gone through",
            "has not gone through",
            "not completed",
            "stuck",
        ),
        "fees_and_pricing": (
            "fee",
            "fees",
            "commission",
            "extra charge",
            "charged for",
            "cost",
            "exchange charge",
            "withdrawal charge",
            "transfer charge",
        ),
    }

    def __init__(self) -> None:
        self.default_queue_: str | None = None

    def fit(self, queues: Sequence[str]) -> "KeywordRuleRouter":
        majority = MajorityQueueRouter().fit(queues)
        self.default_queue_ = majority.queue_
        return self

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    def predict_one(self, text: str) -> str:
        if self.default_queue_ is None:
            raise RuntimeError("KeywordRuleRouter must be fitted before prediction")
        normalised = self._normalise(text)
        scores = {
            queue: sum(len(term.split()) for term in terms if term in normalised)
            for queue, terms in self.KEYWORDS.items()
        }
        best_score = max(scores.values())
        if best_score == 0:
            return self.default_queue_
        winners = sorted(queue for queue, score in scores.items() if score == best_score)
        return winners[0]

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.predict_one(text) for text in texts]


class TfidfIntentRouter:
    def __init__(self, seed: int = 6201) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import FeatureUnion, Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required. Install the project with: python -m pip install -e ."
            ) from exc

        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        self.pipeline = Pipeline([("features", features), ("classifier", classifier)])

    def fit(self, texts: Sequence[str], intents: Sequence[str]) -> "TfidfIntentRouter":
        if len(texts) != len(intents) or not texts:
            raise ValueError("Training texts and intents must have the same non-zero length")
        self.pipeline.fit(list(texts), list(intents))
        return self

    def predict_intents(self, texts: Sequence[str]) -> list[str]:
        return list(self.pipeline.predict(list(texts)))

    def predict_queues(self, texts: Sequence[str]) -> list[str]:
        return [queue_for_intent(intent) for intent in self.predict_intents(texts)]

    def predict_confidence(self, texts: Sequence[str]) -> list[float]:
        probabilities = self.pipeline.predict_proba(list(texts))
        return [float(row.max()) for row in probabilities]

    def save(self, path: Path) -> None:
        from joblib import dump

        path.parent.mkdir(parents=True, exist_ok=True)
        dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfIntentRouter":
        from joblib import load

        instance = cls()
        instance.pipeline = load(path)
        return instance

