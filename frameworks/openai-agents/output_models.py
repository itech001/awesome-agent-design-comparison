"""Structured output types for the OpenAI Agents SDK solver.

This framework realizes the two-loop multi-agent design from
`docs/agents-design.md`:
  - the Resolver produces an answer for one question (ResolverOutput);
  - the Validator independently checks it (ValidatorVerdict);
  - solve_one runs the inner loop (resolve -> validate) up to MAX_ATTEMPTS and
    returns a QuestionStatus.

`AgentAnswer` is retained as the Resolver's structured type and as the contract's
per-answer shape. `normalize_response` maps a raw response onto the contract's
`response` field (a single letter A-D for MC).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

MAX_ATTEMPTS = 3
"""Inner-loop cap before keeping the last answer (see agents-design.md)."""


class AgentAnswer(BaseModel):
    """What an agent returns for one question (response + reasoning)."""

    response: str = Field(
        min_length=1,
        description="The final answer: a single letter for MC, or the full answer text for short-answer questions",
    )
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought leading to the answer",
    )


# Alias for the Resolver's output type. Kept as a distinct name for clarity,
# though it carries the same shape as AgentAnswer.
ResolverOutput = AgentAnswer


class ValidatorVerdict(BaseModel):
    """The Validator's independent verdict on a Resolver's answer."""

    accepted: bool = Field(description="True if the answer is correct.")
    feedback: str = Field(
        default="",
        description="Empty when accepted; the reason for revision otherwise.",
    )
    checked_answer: str = Field(
        description="The Validator's own independently-derived answer.",
    )


class QuestionStatus(BaseModel):
    """Outcome of the inner loop for one question."""

    question_id: str
    final_response: str
    reasoning: str = ""
    accepted: bool = False
    attempts: int = 0
    validator_answer: str | None = None
    latency_ms: int = 0


# A standalone option letter: word boundary, A-D, then optional punctuation/end.
_OPTION_LETTER = re.compile(r"\b([A-Da-d])\b")
# Fallback: the first alphabetic char if no option letter is present.
_FIRST_LETTER = re.compile(r"[A-Za-z]")


def normalize_response(raw: str, *, is_mc: bool) -> str:
    """Normalize a response for the contract's `response` field.

    For multiple choice, extract a single letter A-D. Prefer a standalone
    option letter (e.g. the B in "Option B" or "B. ..."); fall back to the
    first alphabetic char if none is found. For short answer, keep the text
    as-is.
    """
    if is_mc:
        text = raw or ""
        m = _OPTION_LETTER.search(text)
        if m is None:
            m = _FIRST_LETTER.search(text)
        return m.group(0).upper() if m else ""
    return raw or ""
