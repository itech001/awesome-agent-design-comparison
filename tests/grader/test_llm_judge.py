from grader.graders.llm_judge import LLMJudgeGrader, JudgeClient
from grader.models import Question, AnswerResult


class FakeClient(JudgeClient):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _short_q():
    return Question(
        id="math-005",
        subject="math",
        type="short_answer",
        question="Solve 3x-7=2x+5",
        answer="x=12",
        answer_type="text",
        scoring={
            "max_points": 2,
            "method": "rubric",
            "rubric": [
                {"criterion": "uses correct method", "points": 1},
                {"criterion": "correct answer", "points": 1},
            ],
        },
        source="s",
        difficulty="easy",
    )


def _answer(text):
    return AnswerResult(
        question_id="math-005", subject="math", type="short_answer",
        response=text, reasoning="", latency_ms=10, raw={},
    )


def test_llm_judge_full_credit():
    client = FakeClient(["1", "1"])  # one verdict per criterion
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("used correct method, answer x=12"))
    assert s.earned == 2
    assert s.correct is True
    assert s.grader == "llm"
    assert len(client.calls) == 2


def test_llm_judge_zero_credit():
    client = FakeClient(["0", "0"])
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("nonsense"))
    assert s.earned == 0
    assert s.correct is False


def test_llm_judge_clamps_invalid_verdict_to_zero():
    # valid verdict passes through (1); non-numeric clamps to 0 -> total 1
    client = FakeClient(["1", "garbage"])
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("x"))
    assert s.earned == 1
