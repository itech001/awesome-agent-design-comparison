"""Build the StateGraph: resolve -> validate -> (conditional) resolve | END.

This is LangGraph's idiomatic expression of the two-loop design from
docs/agents-design.md. The conditional edge after `validate` is the
validate->revise decision: END if accepted or attempts capped, else back to
`resolve`. That edge — not an imperative while-loop — is what makes this the
LangGraph showcase.
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from output_models import (
    MAX_ATTEMPTS,
    normalize_response,
    parse_final_text,
    parse_validator_text,
)
from state import State


def _build_question_prompt(q: dict[str, Any]) -> str:
    is_mc = q.get("type") == "multiple_choice"
    kind = "multiple choice" if is_mc else "short answer"
    lines = [
        f"Subject: {q.get('subject', '')}",
        f"Question type: {kind}",
        "",
        q.get("question", ""),
    ]
    if is_mc and q.get("options"):
        lines += ["", "Options:"] + [f"  {o}" for o in q["options"]]
    return "\n".join(lines)


def _build_resolver_prompt(q: dict[str, Any], feedback: str = "") -> str:
    base = _build_question_prompt(q)
    suffix = (
        "Remember: respond with a single capital letter for multiple choice, "
        "or the complete answer for short answer."
    )
    if feedback:
        return (
            f"{base}\n\n"
            "A previous attempt at this question was rejected. Revise your answer.\n"
            f"Examiner feedback: {feedback}\n\n"
            + suffix
        )
    return f"{base}\n\n{suffix}"


def _build_validator_prompt(q: dict[str, Any], student_answer: str) -> str:
    base = _build_question_prompt(q)
    return (
        f"{base}\n\n"
        f"Student's answer to verify: {student_answer}\n\n"
        "Solve the question yourself, then decide whether the student's answer "
        "is correct. For multiple choice, `checked_answer` must be a single "
        "capital letter; for short answer, the complete correct answer."
    )


def _make_resolve_node(invoke_resolve: Callable[[str], str]):
    def resolve(state: State) -> dict:
        q = state["question"]
        prompt = _build_resolver_prompt(q, state.get("feedback", ""))
        raw_text = invoke_resolve(prompt)
        parsed = parse_final_text(raw_text, is_mc=state["is_mc"])
        return {
            "raw_response": parsed.response,
            "response": normalize_response(parsed.response, is_mc=state["is_mc"]),
            "reasoning": parsed.reasoning,
            "attempts": state.get("attempts", 0) + 1,
        }

    return resolve


def _make_validate_node(invoke_validate: Callable[[str], str]):
    def validate(state: State) -> dict:
        raw_text = invoke_validate(
            _build_validator_prompt(state["question"], state["raw_response"])
        )
        verdict = parse_validator_text(raw_text, is_mc=state["is_mc"])
        return {
            "accepted": verdict.accepted,
            "feedback": verdict.feedback,
            "validator_answer": normalize_response(verdict.checked_answer, is_mc=state["is_mc"]),
        }

    return validate


def _route_after_validate(state: State) -> str:
    """Conditional edge: END if accepted or capped, else back to resolve."""
    attempts = state.get("attempts", 0)
    max_attempts = state.get("max_attempts", MAX_ATTEMPTS)
    if state.get("accepted") or attempts >= max_attempts:
        return END
    return "resolve"


def build_graph(
    *,
    invoke_resolve: Callable[[str], str],
    invoke_validate: Callable[[str], str],
    enable_validator: bool = True,
):
    """Compile the resolve/validate StateGraph.

    `invoke_resolve` / `invoke_validate` are callables (prompt) -> final text,
    typically wrapping the same or two different LLMs. Injected so tests run
    without API calls.
    """
    builder = StateGraph(State)
    builder.add_node("resolve", _make_resolve_node(invoke_resolve))
    if enable_validator:
        builder.add_node("validate", _make_validate_node(invoke_validate))
        builder.add_edge(START, "resolve")
        builder.add_edge("resolve", "validate")
        # path_map avoids LangGraph issue #987 (edges to every node).
        builder.add_conditional_edges(
            "validate", _route_after_validate, {"resolve": "resolve", END: END}
        )
    else:
        builder.add_edge(START, "resolve")
        builder.add_edge("resolve", END)
    return builder.compile()
