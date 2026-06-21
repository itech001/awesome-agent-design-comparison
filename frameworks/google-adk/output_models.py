"""Structured output schema for the Google ADK solver.

`AgentAnswer` is passed as the LlmAgent's `output_schema`. The Runner's final
response text is the JSON-serialized schema; `parse_final_text` parses it back
(and tolerates plain-text responses). `normalize_response` maps the response
onto the contract's `response` field (single A-D letter for MC).
"""
from __future__ import annotations

import json
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
    """Parse the Runner's final response text into an AgentAnswer.

    ADK with output_schema returns JSON matching the schema. If parsing fails
    (model returned plain text), fall back to treating the whole text as the
    response with empty reasoning. Raises ValueError if the text yields no
    usable response (caller should record an empty-response error).
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
    # Plain-text fallback: treat as the raw response, no reasoning.
    return AgentAnswer(response=text, reasoning="")
