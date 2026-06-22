"""Tests for the two-loop (Resolver -> Validator) multi-agent design.

All tests inject fake runners so no API calls are made.
"""
from output_models import AgentAnswer, ValidatorVerdict
from solver import solve_one, get_status


def _mc_question():
    return {
        "id": "math-001",
        "subject": "math",
        "type": "multiple_choice",
        "question": "Simplify (x+2)(x-2).",
        "options": ["A. x^2-4", "B. x^2+4", "C. x^2-2x", "D. x^2-4x+4"],
        "answer": "A",
        "answer_type": "letter",
    }


class ScriptedResolver:
    """Returns each scripted ResolverOutput in turn."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = []

    def __call__(self, agent, prompt):
        self.calls.append(prompt)
        return self._answers.pop(0)


class ScriptedValidator:
    """Returns each scripted ValidatorVerdict in turn."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = []

    def __call__(self, agent, prompt):
        self.calls.append(prompt)
        return self._verdicts.pop(0)


def test_two_loop_accepts_on_first_try():
    resolver = ScriptedResolver([AgentAnswer(response="A", reasoning="diff of squares")])
    validator = ScriptedValidator([
        ValidatorVerdict(accepted=True, checked_answer="A"),
    ])
    answer, latency = solve_one(
        _mc_question(),
        resolver_runner=resolver,
        validator_runner=validator,
    )
    assert answer.response == "A"
    assert latency >= 0
    status = get_status(answer)
    assert status.accepted is True
    assert status.attempts == 1
    assert status.validator_answer == "A"
    assert len(resolver.calls) == 1
    assert len(validator.calls) == 1


def test_two_loop_revises_then_accepts():
    resolver = ScriptedResolver([
        AgentAnswer(response="B", reasoning="guessed"),
        AgentAnswer(response="A", reasoning="reconsidered"),
    ])
    validator = ScriptedValidator([
        ValidatorVerdict(accepted=False, feedback="B is wrong; reconsider", checked_answer="A"),
        ValidatorVerdict(accepted=True, checked_answer="A"),
    ])
    answer, _ = solve_one(
        _mc_question(),
        resolver_runner=resolver,
        validator_runner=validator,
    )
    assert answer.response == "A"
    status = get_status(answer)
    assert status.accepted is True
    assert status.attempts == 2
    # The second resolver call should carry the validator's feedback.
    assert "B is wrong" in resolver.calls[1]


def test_two_loop_caps_at_max_attempts():
    # Never accepts: should run exactly max_attempts times.
    max_attempts = 2
    resolver = ScriptedResolver([
        AgentAnswer(response="B", reasoning=""),
        AgentAnswer(response="C", reasoning=""),
    ])
    validator = ScriptedValidator([
        ValidatorVerdict(accepted=False, feedback="no", checked_answer="A"),
        ValidatorVerdict(accepted=False, feedback="still no", checked_answer="A"),
    ])
    answer, _ = solve_one(
        _mc_question(),
        resolver_runner=resolver,
        validator_runner=validator,
        max_attempts=max_attempts,
    )
    status = get_status(answer)
    assert status.accepted is False
    assert status.attempts == max_attempts
    assert answer.response == "C"  # last attempt kept


def test_two_loop_disabled_runs_resolver_once():
    resolver = ScriptedResolver([AgentAnswer(response="A", reasoning="r")])
    validator = ScriptedValidator([])  # must not be called
    answer, _ = solve_one(
        _mc_question(),
        resolver_runner=resolver,
        validator_runner=validator,
        enable_validator=False,
    )
    assert answer.response == "A"
    status = get_status(answer)
    assert status.attempts == 1
    assert status.validator_answer is None
    assert len(validator.calls) == 0


def test_get_status_returns_none_when_not_attached():
    bare = AgentAnswer(response="A", reasoning="")
    assert get_status(bare) is None
