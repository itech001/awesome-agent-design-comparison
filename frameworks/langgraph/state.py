"""The graph State. Carries everything across the resolve/validate nodes."""
from __future__ import annotations

from typing import TypedDict


class State(TypedDict, total=False):
    """Shared state for the resolve -> validate graph.

    `total=False` so the initial invoke can pass just the required setup keys;
    nodes add the rest as they run.
    """

    # setup (provided at invoke time)
    question: dict          # the dataset question
    is_mc: bool
    enable_validator: bool
    max_attempts: int

    # resolve outputs (updated by the resolve node)
    response: str           # current resolver answer (normalized)
    reasoning: str
    raw_response: str       # pre-normalization, for the validator prompt

    # loop bookkeeping
    attempts: int
    feedback: str           # validator's revision note (fed back to resolve)

    # validate outputs (updated by the validate node)
    accepted: bool
    validator_answer: str
