from solver import solve_one, QuestionInput, build_prompt
from output_models import AgentAnswer


class FakeRunner:
    def __init__(self, answer: AgentAnswer):
        self._answer = answer
        self.calls = []

    def __call__(self, agent, prompt):
        self.calls.append((agent, prompt))
        return self._answer


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


def _short_question():
    return {
        "id": "math-005",
        "subject": "math",
        "type": "short_answer",
        "question": "Solve 3x - 7 = 2x + 5 for x.",
        "answer": "x = 12",
        "answer_type": "text",
    }


def test_build_prompt_includes_question_and_options():
    p = build_prompt(_mc_question())
    assert "Simplify (x+2)(x-2)." in p
    assert "A. x^2-4" in p
    assert "multiple choice" in p.lower()


def test_build_prompt_short_answer_no_options():
    p = build_prompt(_short_question())
    assert "Solve 3x - 7 = 2x + 5" in p
    assert "short answer" in p.lower()


def test_solve_one_mc_returns_normalized_letter():
    fake = FakeRunner(AgentAnswer(response="A", reasoning="diff of squares"))
    answer, latency = solve_one(_mc_question(), runner=fake)
    assert answer.response == "A"
    assert latency >= 0
    assert len(fake.calls) == 1


def test_solve_one_mc_normalizes_to_letter():
    fake = FakeRunner(AgentAnswer(response="Option B", reasoning="..."))
    answer, _ = solve_one(_mc_question(), runner=fake)
    assert answer.response == "B"


def test_solve_one_short_keeps_verbatim():
    fake = FakeRunner(AgentAnswer(response="x = 12", reasoning="subtracted 2x"))
    answer, _ = solve_one(_short_question(), runner=fake)
    assert answer.response == "x = 12"


def test_question_input_parsing():
    qi = QuestionInput.from_question(_mc_question())
    assert qi.id == "math-001"
    assert qi.is_mc is True
    assert qi.options == _mc_question()["options"]
