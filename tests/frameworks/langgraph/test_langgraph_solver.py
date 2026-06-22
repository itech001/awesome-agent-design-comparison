"""Solver tests with injectable fake LLMs (no API calls)."""
from solver import solve_one, get_status
from output_models import MAX_ATTEMPTS


class FakeLLM:
    """Returns scripted text per call, separate queues for resolve/validate.

    Distinguishes the two by inspecting the prompt for 'verify' (validator).
    """

    def __init__(self, resolve_texts, validate_texts):
        self._r = list(resolve_texts)
        self._v = list(validate_texts)
        self.resolve_calls = []
        self.validate_calls = []

    def invoke(self, prompt: str) -> str:
        if "verify" in prompt.lower():
            self.validate_calls.append(prompt)
            return self._v.pop(0)
        self.resolve_calls.append(prompt)
        return self._r.pop(0)

    # expose .content-compatible: tests return plain str, so no wrapper needed


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


def test_solve_one_accepts_on_first_try():
    llm = FakeLLM(
        ['{"response": "A", "reasoning": "r"}'],
        ['{"accepted": true, "feedback": "", "checked_answer": "A"}'],
    )
    answer, latency = solve_one(_mc_question(), llm=llm)
    assert answer.response == "A"
    assert latency >= 0
    status = get_status(answer)
    assert status.accepted is True
    assert status.attempts == 1
    assert status.validator_answer == "A"


def test_solve_one_revises_then_accepts():
    llm = FakeLLM(
        ['{"response": "B", "reasoning": ""}', '{"response": "A", "reasoning": ""}'],
        [
            '{"accepted": false, "feedback": "wrong", "checked_answer": "A"}',
            '{"accepted": true, "feedback": "", "checked_answer": "A"}',
        ],
    )
    answer, _ = solve_one(_mc_question(), llm=llm)
    assert answer.response == "A"
    status = get_status(answer)
    assert status.accepted is True
    assert status.attempts == 2


def test_solve_one_caps_at_max_attempts():
    llm = FakeLLM(
        ['{"response": "B", "reasoning": ""}', '{"response": "C", "reasoning": ""}'],
        [
            '{"accepted": false, "feedback": "no", "checked_answer": "A"}',
            '{"accepted": false, "feedback": "still no", "checked_answer": "A"}',
        ],
    )
    answer, _ = solve_one(_mc_question(), llm=llm, max_attempts=2)
    status = get_status(answer)
    assert status.accepted is False
    assert status.attempts == 2
    assert answer.response == "C"


def test_solve_one_disabled_validator():
    llm = FakeLLM(
        ['{"response": "A", "reasoning": "r"}'],
        [],  # validator must not be called
    )
    answer, _ = solve_one(_mc_question(), llm=llm, enable_validator=False)
    assert answer.response == "A"
    status = get_status(answer)
    assert status.attempts == 1
    assert status.validator_answer is None
    assert len(llm.validate_calls) == 0
