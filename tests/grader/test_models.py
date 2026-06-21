import pytest
from pydantic import ValidationError

from grader.models import Question, AnswerResult, ResultFile, Score


def test_question_mc_ok():
    q = Question(
        id="math-001",
        subject="math",
        type="multiple_choice",
        question="q",
        options=["A. x", "B. y"],
        answer="A",
        answer_type="letter",
        scoring={"max_points": 1, "method": "exact"},
        source="s",
        difficulty="easy",
    )
    assert q.scoring.max_points == 1


def test_question_rejects_bad_subject():
    with pytest.raises(ValidationError):
        Question(
            id="math-001",
            subject="art",
            type="multiple_choice",
            question="q",
            options=["A", "B"],
            answer="A",
            answer_type="letter",
            scoring={"max_points": 1, "method": "exact"},
            source="s",
            difficulty="easy",
        )


def test_answer_result_ok():
    a = AnswerResult(
        question_id="math-001",
        subject="math",
        type="multiple_choice",
        response="A",
        reasoning="",
        latency_ms=100,
        raw={},
    )
    assert a.response == "A"


def test_result_file_ok():
    rf = ResultFile(
        framework="test",
        model="m",
        generated_at="2026-06-21T10:00:00Z",
        config={},
        results=[],
    )
    assert rf.framework == "test"


def test_score_normalized():
    s = Score(
        question_id="x",
        earned=2,
        max_points=4,
        normalized=0.5,
        correct=False,
        per_criterion=[],
        grader="rubric",
    )
    assert s.normalized == 0.5
