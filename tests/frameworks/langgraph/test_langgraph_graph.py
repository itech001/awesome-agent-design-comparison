"""Tests for the StateGraph: the idiomatic two-loop core.

Injects fake invoke_resolve / invoke_validate callables so no API calls are made.
"""
from graph import build_graph, State


def _initial_state(enable_validator=True, max_attempts=3):
    return {
        "question": {
            "id": "math-001",
            "subject": "math",
            "type": "multiple_choice",
            "question": "Simplify (x+2)(x-2).",
            "options": ["A. x^2-4", "B. x^2+4", "C. x^2-2x", "D. x^2-4x+4"],
        },
        "is_mc": True,
        "enable_validator": enable_validator,
        "max_attempts": max_attempts,
    }


class Scripted:
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return self._texts.pop(0)


def test_graph_accepts_on_first_try():
    r = Scripted(['{"response": "A", "reasoning": "r"}'])
    v = Scripted(['{"accepted": true, "feedback": "", "checked_answer": "A"}'])
    g = build_graph(invoke_resolve=r, invoke_validate=v)
    final = g.invoke(_initial_state())
    assert final["accepted"] is True
    assert final["attempts"] == 1
    assert final["response"] == "A"
    assert len(r.calls) == 1
    assert len(v.calls) == 1


def test_graph_revises_then_accepts():
    r = Scripted([
        '{"response": "B", "reasoning": ""}',
        '{"response": "A", "reasoning": ""}',
    ])
    v = Scripted([
        '{"accepted": false, "feedback": "wrong", "checked_answer": "A"}',
        '{"accepted": true, "feedback": "", "checked_answer": "A"}',
    ])
    g = build_graph(invoke_resolve=r, invoke_validate=v)
    final = g.invoke(_initial_state())
    assert final["accepted"] is True
    assert final["attempts"] == 2
    assert final["response"] == "A"
    # The second resolver call carries the feedback.
    assert "wrong" in r.calls[1]


def test_graph_caps_at_max_attempts():
    r = Scripted([
        '{"response": "B", "reasoning": ""}',
        '{"response": "C", "reasoning": ""}',
    ])
    v = Scripted([
        '{"accepted": false, "feedback": "no", "checked_answer": "A"}',
        '{"accepted": false, "feedback": "still no", "checked_answer": "A"}',
    ])
    g = build_graph(invoke_resolve=r, invoke_validate=v)
    final = g.invoke(_initial_state(max_attempts=2))
    assert final["accepted"] is False
    assert final["attempts"] == 2
    assert final["response"] == "C"


def test_graph_disabled_validator_runs_resolve_once():
    r = Scripted(['{"response": "A", "reasoning": "r"}'])
    v = Scripted([])  # must not be called
    g = build_graph(invoke_resolve=r, invoke_validate=v, enable_validator=False)
    final = g.invoke(_initial_state(enable_validator=False))
    assert final["response"] == "A"
    assert final["attempts"] == 1
    assert "accepted" not in final  # validate never ran
    assert len(v.calls) == 0
