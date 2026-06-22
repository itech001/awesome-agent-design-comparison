"""Solve one question by compiling + invoking the resolve/validate graph.

Wraps the graph from graph.py. The LLM (or fakes) are injectable so tests run
without API calls.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from graph import build_graph
from llm import LLM, build_llm
from output_models import (
    AgentAnswer,
    MAX_ATTEMPTS,
    QuestionStatus,
)


def _make_invoke(llm: LLM, system_hint: str) -> Callable[[str], str]:
    """Wrap an LLM so invoke(prompt) returns the final text."""

    def invoke(prompt: str) -> str:
        # langchain ChatOpenAI.invoke takes a string and returns an AIMessage;
        # .content is the text. A plain fake just returns str.
        resp = llm.invoke(prompt)
        return getattr(resp, "content", resp)

    return invoke


def solve_one(
    q: dict[str, Any],
    *,
    llm: LLM | None = None,
    resolver_llm: LLM | None = None,
    validator_llm: LLM | None = None,
    enable_validator: bool = True,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (final AgentAnswer, latency_ms).

    By default both nodes use the same LLM (build_llm()). Pass resolver_llm /
    validator_llm to use different models, or `llm` for tests to inject a fake
    used by both nodes. enable_validator=False runs resolve once (baseline).
    """
    is_mc = q.get("type") == "multiple_choice"
    r_llm = resolver_llm or llm or build_llm()
    v_llm = validator_llm or llm or build_llm()
    invoke_resolve = _make_invoke(r_llm, "resolver")
    invoke_validate = _make_invoke(v_llm, "validator")

    graph = build_graph(
        invoke_resolve=invoke_resolve,
        invoke_validate=invoke_validate,
        enable_validator=enable_validator,
    )

    initial = {
        "question": q,
        "is_mc": is_mc,
        "enable_validator": enable_validator,
        "max_attempts": max_attempts,
    }

    start = time.perf_counter()
    final = graph.invoke(initial)
    latency_ms = int((time.perf_counter() - start) * 1000)

    response = final.get("response", "")
    reasoning = final.get("reasoning", "")
    # If resolve produced nothing usable, fall back to a non-empty placeholder
    # so AgentAnswer's min_length=1 is satisfied; run.py records the failure.
    if not response:
        response = "?"

    answer = AgentAnswer(response=response, reasoning=reasoning)
    status = QuestionStatus(
        question_id=q["id"],
        final_response=response,
        reasoning=reasoning,
        accepted=final.get("accepted", True) if enable_validator else True,
        attempts=final.get("attempts", 1),
        validator_answer=final.get("validator_answer") if enable_validator else None,
        latency_ms=latency_ms,
    )
    answer.__status__ = status  # type: ignore[attr-defined]
    return answer, latency_ms


def get_status(answer: AgentAnswer) -> QuestionStatus | None:
    """Read the QuestionStatus attached by solve_one, if any."""
    return getattr(answer, "__status__", None)
