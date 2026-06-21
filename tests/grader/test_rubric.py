from grader.graders.rubric import RubricGrader
from grader.models import Question, AnswerResult


def _short_q(rubric, notes=""):
    max_points = sum(c["points"] for c in rubric)
    return Question(
        id="math-005",
        subject="math",
        type="short_answer",
        question="Solve 3x-7=2x+5",
        answer="x=12",
        answer_type="text",
        scoring={
            "max_points": max_points,
            "method": "rubric",
            "rubric": rubric,
        },
        source="s",
        difficulty="easy",
        notes=notes,
    )


def _answer(response):
    return AnswerResult(
        question_id="math-005",
        subject="math",
        type="short_answer",
        response=response,
        reasoning="",
        latency_ms=10,
        raw={},
    )


def test_rubric_full_credit():
    q = _short_q([
        {"criterion": "moves variable terms to one side", "points": 1},
        {"criterion": "moves constants to the other side", "points": 1},
        {"criterion": "correct final answer", "points": 1},
    ])
    a = _answer("First I moved variable terms to one side, then moved constants to the other side, getting the correct final answer x=12.")
    s = RubricGrader().score(q, a)
    assert s.earned == 3
    assert s.correct is True
    assert s.normalized == 1.0
    assert len(s.per_criterion) == 3


def test_rubric_partial_credit():
    q = _short_q([
        {"criterion": "uses pythagorean theorem", "points": 1},
        {"criterion": "correct answer 5", "points": 1},
    ])
    a = _answer("I used the pythagorean theorem but got the wrong number.")
    s = RubricGrader().score(q, a)
    assert s.earned == 1
    assert s.correct is False
    assert s.normalized == 0.5


def test_rubric_no_credit():
    q = _short_q([{"criterion": "mentions photosynthesis", "points": 1}])
    a = _answer("The sky is blue.")
    s = RubricGrader().score(q, a)
    assert s.earned == 0
    assert s.correct is False


def test_rubric_uses_notes_for_synonyms():
    q = _short_q(
        [{"criterion": "correct answer 5", "points": 1}],
        notes="accept: five; sqrt(25)",
    )
    a = _answer("The answer is five.")
    s = RubricGrader().score(q, a)
    assert s.earned == 1
