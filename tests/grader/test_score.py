from pathlib import Path

from grader.score import score_result_file, aggregate
from grader.loaders import load_result_file

_FIXTURE = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "sample_result.json"
_DATASET = Path(__file__).resolve().parent.parent.parent / "dataset" / "questions.json"


def test_score_result_file_returns_per_question_scores():
    rf = load_result_file(_FIXTURE)
    scored = score_result_file(rf, dataset_path=_DATASET)
    # fixture has 2 answers; both should be scored
    assert len(scored) == 2
    # math-001 answer is A, correct
    by_id = {s.question_id: s for s in scored}
    assert by_id["math-001"].correct is True


def test_score_result_file_missing_question_skipped():
    # Build a result file with an unknown question_id; should be skipped, not crash.
    from grader.models import ResultFile, AnswerResult
    rf = ResultFile(
        framework="x", model="m", generated_at="2026-06-21T10:00:00Z",
        config={}, results=[AnswerResult(
            question_id="math-999", subject="math", type="multiple_choice",
            response="A", reasoning="", latency_ms=1, raw={},
        )],
    )
    scored = score_result_file(rf, dataset_path=_DATASET)
    assert scored == []


def test_aggregate_computes_overall_and_by_subject():
    rf = load_result_file(_FIXTURE)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    assert 0.0 <= agg["overall"] <= 1.0
    assert "by_subject" in agg
    assert "by_type" in agg
    assert "math" in agg["by_subject"]
