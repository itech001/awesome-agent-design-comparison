from pathlib import Path

from grader.report import render_per_framework, render_comparison
from grader.loaders import load_result_file
from grader.score import score_result_file, aggregate

_FIXTURE = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "sample_result.json"
_DATASET = Path(__file__).resolve().parent.parent.parent / "dataset" / "questions.json"


def test_render_per_framework_produces_markdown():
    rf = load_result_file(_FIXTURE)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    md = render_per_framework(rf.framework, rf.model, agg, scored)
    assert "# " in md
    assert "Overall" in md
    assert "math-001" in md


def test_render_comparison_produces_table():
    summaries = [{"framework": "sample", "model": "m", "overall": 0.9,
                  "by_subject": {"math": 0.9}, "by_type": {"multiple_choice": 1.0}, "count": 2}]
    md = render_comparison(summaries)
    assert "| framework |" in md
    assert "sample" in md
