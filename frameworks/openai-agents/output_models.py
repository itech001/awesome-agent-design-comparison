"""Structured output type for the OpenAI Agents SDK solver.

`AgentAnswer` is passed as the agent's `output_type`, so the SDK returns a
parsed instance via `result.final_output`. `normalize_response` maps the raw
response onto the contract's `response` field (a single letter for MC).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


class AgentAnswer(BaseModel):
    """What the agent returns for one question."""

    response: str = Field(
        min_length=1,
        description="The final answer: a single letter for MC, or the full answer text for short-answer questions",
    )
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought leading to the answer",
    )


# A standalone option letter: word boundary, A-D, then optional punctuation/end.
_OPTION_LETTER = re.compile(r"\b([A-Da-d])\b")
# Fallback: the first alphabetic char if no option letter is present.
_FIRST_LETTER = re.compile(r"[A-Za-z]")


def normalize_response(raw: str, *, is_mc: bool) -> str:
    """Normalize the agent's response for the contract's `response` field.

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
