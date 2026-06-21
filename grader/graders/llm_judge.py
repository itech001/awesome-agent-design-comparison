"""Optional LLM-as-judge grader. Off by default; --judge llm enables it.

The LLM client is injected so tests stay deterministic and free.
"""
from __future__ import annotations

import abc
import re

from grader.graders.base import Grader
from grader.models import AnswerResult, CriterionScore, Question, Score


class JudgeClient(abc.ABC):
    """Abstract LLM client. judge(prompt) returns the model's text response."""

    @abc.abstractmethod
    def judge(self, prompt: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class OpenAIJudgeClient(JudgeClient):
    """Real client backed by the OpenAI API. Lazy import so tests don't need it."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def judge(self, prompt: str) -> str:
        import os
        from openai import OpenAI  # local import: optional dependency
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )
        return resp.output_text.strip()


def _verdict_to_int(text: str, available: int) -> int:
    """Extract an integer verdict from the model's response; clamp to [0, available]."""
    m = re.search(r"\d+", text or "")
    if not m:
        return 0
    val = int(m.group(0))
    return max(0, min(available, val))


class LLMJudgeGrader(Grader):
    name = "llm"

    def __init__(self, client: JudgeClient):
        self.client = client

    def _prompt(self, question: Question, answer: AnswerResult, criterion: str, available: int) -> str:
        return (
            f"You are grading a Zhongkao exam answer.\n"
            f"Question: {question.question}\n"
            f"Reference answer: {question.answer}\n"
            f"Student response: {answer.response}\n\n"
            f"Criterion: {criterion} (worth {available} point(s)).\n"
            f"Reply with a single integer from 0 to {available}: "
            f"how many points the student earned for THIS criterion only."
        )

    def score(self, question: Question, answer: AnswerResult) -> Score:
        per: list[CriterionScore] = []
        earned = 0
        max_points = question.scoring.max_points
        for crit in question.scoring.rubric or []:
            prompt = self._prompt(question, answer, crit.criterion, crit.points)
            raw = self.client.judge(prompt)
            got = _verdict_to_int(raw, crit.points)
            earned += got
            per.append(CriterionScore(criterion=crit.criterion, earned=got, available=crit.points))
        correct = earned == max_points and max_points > 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=per,
            grader=self.name,
        )
