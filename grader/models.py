"""Pydantic models for questions, answer results, result files, and scores."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Subject = Literal[
    "chinese", "math", "english", "physics",
    "chemistry", "biology", "history", "geography",
]
QType = Literal["multiple_choice", "short_answer"]
AnswerType = Literal["letter", "text"]
Method = Literal["exact", "rubric"]


class RubricCriterion(BaseModel):
    criterion: str
    points: int = Field(ge=1)


class Scoring(BaseModel):
    max_points: int = Field(ge=1)
    method: Method
    rubric: list[RubricCriterion] | None = None


class Question(BaseModel):
    id: str
    subject: Subject
    type: QType
    question: str
    options: list[str] | None = None
    answer: str
    answer_type: AnswerType
    scoring: Scoring
    source: str
    difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


class AnswerResult(BaseModel):
    question_id: str
    subject: Subject
    type: QType
    response: str
    reasoning: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class ResultFile(BaseModel):
    framework: str
    model: str
    generated_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[AnswerResult]


class CriterionScore(BaseModel):
    criterion: str
    earned: int
    available: int


class Score(BaseModel):
    question_id: str
    earned: float
    max_points: int
    normalized: float
    correct: bool
    per_criterion: list[CriterionScore]
    grader: str  # which grader produced this ("exact", "rubric", "llm")
