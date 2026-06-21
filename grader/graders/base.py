"""Abstract base for graders."""
from __future__ import annotations

import abc

from grader.models import AnswerResult, Question, Score


class Grader(abc.ABC):
    """A grader scores a single (question, answer) pair."""

    name: str = "base"

    @abc.abstractmethod
    def score(self, question: Question, answer: AnswerResult) -> Score:  # pragma: no cover - abstract
        raise NotImplementedError
