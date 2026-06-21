from grader.graders.exact import ExactGrader
from grader.models import Question, AnswerResult


def _mc_question(answer="A"):
    return Question(
        id="math-001",
        subject="math",
        type="multiple_choice",
        question="q",
        options=["A. x", "B. y"],
        answer=answer,
        answer_type="letter",
        scoring={"max_points": 1, "method": "exact"},
        source="s",
        difficulty="easy",
    )


def _answer(response="A"):
    return AnswerResult(
        question_id="math-001",
        subject="math",
        type="multiple_choice",
        response=response,
        reasoning="",
        latency_ms=10,
        raw={},
    )


def test_exact_correct():
    s = ExactGrader().score(_mc_question("A"), _answer("A"))
    assert s.earned == 1
    assert s.correct is True
    assert s.normalized == 1.0
    assert s.grader == "exact"


def test_exact_wrong():
    s = ExactGrader().score(_mc_question("A"), _answer("B"))
    assert s.earned == 0
    assert s.correct is False
    assert s.normalized == 0.0


def test_exact_case_insensitive():
    s = ExactGrader().score(_mc_question("A"), _answer("a"))
    assert s.correct is True


def test_exact_strips_response():
    s = ExactGrader().score(_mc_question("A"), _answer(" A. "))
    assert s.correct is True
