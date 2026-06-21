"""Exact-match grader for multiple-choice (answer_type=letter)."""
from __future__ import annotations

import re

from grader.graders.base import Grader
from grader.models import AnswerResult, Question, Score


def _extract_letter(response: str) -> str:
    """Pull the first A-Z letter from the response, case-insensitive."""
    m = re.search(r"[A-Za-z]", response or "")
    return m.group(0).upper() if m else ""


class ExactGrader(Grader):
    name = "exact"

    def score(self, question: Question, answer: AnswerResult) -> Score:
        expected = question.answer.strip().upper()
        got = _extract_letter(answer.response)
        correct = got == expected
        max_points = question.scoring.max_points
        earned = max_points if correct else 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=[],
            grader=self.name,
        )
