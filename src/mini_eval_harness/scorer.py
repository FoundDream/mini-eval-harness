from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class Score:
    score: float
    correct: bool
    details: dict[str, Any] = field(default_factory=dict)


class ExactMatchScorer:
    name = "exact_match"

    def score(self, prediction: str, gold: str) -> Score:
        normalized_prediction = self._normalize(prediction)
        normalized_gold = self._normalize(gold)
        correct = normalized_prediction == normalized_gold
        return Score(
            score=1.0 if correct else 0.0,
            correct=correct,
            details={
                "normalized_prediction": normalized_prediction,
                "normalized_gold": normalized_gold,
                "error_type": None if correct else "exact_match_mismatch",
            },
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()


class GSM8KFinalAnswerScorer:
    name = "gsm8k_final_answer"

    def score(self, prediction: str, gold: str) -> Score:
        predicted_answer = self._extract_answer(prediction)
        gold_answer = self._extract_answer(gold)

        if gold_answer is None:
            return Score(
                score=0.0,
                correct=False,
                details={
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "error_type": "no_gold_numeric_answer",
                },
            )

        if predicted_answer is None:
            return Score(
                score=0.0,
                correct=False,
                details={
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "error_type": "no_predicted_numeric_answer",
                },
            )

        correct = predicted_answer == gold_answer
        return Score(
            score=1.0 if correct else 0.0,
            correct=correct,
            details={
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "error_type": None if correct else "wrong_numeric_answer",
            },
        )

    @classmethod
    def _extract_answer(cls, text: str) -> str | None:
        answer_region = text.split("####")[-1] if "####" in text else text
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_region)
        if not matches:
            return None
        return cls._normalize_number(matches[-1])

    @staticmethod
    def _normalize_number(value: str) -> str | None:
        cleaned = value.replace(",", "").strip()
        try:
            decimal_value = Decimal(cleaned)
        except InvalidOperation:
            return None

        if decimal_value == decimal_value.to_integral_value():
            return str(decimal_value.to_integral_value())
        return format(decimal_value.normalize(), "f")
