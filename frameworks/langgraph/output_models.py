"""Structured output schemas for the LangGraph solver.

This framework realizes the two-loop multi-agent design from
`docs/agents-design.md` as an explicit StateGraph: a `resolve` node produces an
answer, a `validate` node checks it, and a conditional edge loops back to
`resolve` on REVISE or terminates on ACCEPT / max-attempts.

LangGraph nodes return dicts that update State; the models here parse the LLM's
text output within nodes.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

MAX_ATTEMPTS = 3
"""Inner-loop cap before keeping the last answer (see agents-design.md)."""


class AgentAnswer(BaseModel):
    """What the Resolver returns for one question."""

    response: str = Field(
        min_length=1,
        description="The final answer: a single letter for MC, or the full answer text for short-answer questions",
    )
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought leading to the answer",
    )


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


_OPTION_LETTER = re.compile(r"\b([A-Da-d])\b")
_FIRST_LETTER = re.compile(r"[A-Za-z]")


def normalize_response(raw: str, *, is_mc: bool) -> str:
    """Normalize the response for the contract's `response` field."""
    if is_mc:
        text = raw or ""
        m = _OPTION_LETTER.search(text)
        if m is None:
            m = _FIRST_LETTER.search(text)
        return m.group(0).upper() if m else ""
    return raw or ""


def parse_final_text(text: str, *, is_mc: bool) -> AgentAnswer:
    """Parse the Resolver's final response text into an AgentAnswer.

    Tolerates plain text. Raises ValueError on empty/unusable output.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty final response text")
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("response"):
            return AgentAnswer(
                response=str(data["response"]),
                reasoning=str(data.get("reasoning", "")),
            )
    except json.JSONDecodeError:
        pass
    return AgentAnswer(response=text, reasoning="")


def parse_validator_text(text: str, *, is_mc: bool) -> ValidatorVerdict:
    """Parse the Validator's final response text into a ValidatorVerdict.

    Tolerates plain text: if JSON parsing fails, treats the text as the
    `checked_answer` and infers `accepted` from keywords.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty validator response text")
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "accepted" in data and data.get("checked_answer"):
            return ValidatorVerdict(
                accepted=bool(data["accepted"]),
                feedback=str(data.get("feedback", "")),
                checked_answer=str(data["checked_answer"]),
            )
    except json.JSONDecodeError:
        pass
    lower = text.lower()
    accepted = any(k in lower for k in ("accept", "correct", "yes", "right"))
    return ValidatorVerdict(
        accepted=accepted,
        feedback="" if accepted else "rejected by validator",
        checked_answer=text,
    )
