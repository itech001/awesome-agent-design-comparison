"""End-to-end: framework output -> shared grader -> report.

Uses a fake runner that returns correct-ish answers so we exercise the full
pipeline without API calls.
"""
import sys
from pathlib import Path

import run as run_module
from output_models import AgentAnswer
from grader.loaders import load_result_file
from grader.score import aggregate, score_result_file
from grader.report import render_per_framework

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"


class CorrectFakeRunner:
    """Returns a fixed correct-ish answer for any question."""

    def __call__(self, agent, prompt):
        return AgentAnswer(response="A", reasoning="reasoning")


def test_framework_output_grades_without_error(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), runner=CorrectFakeRunner())
    rf = load_result_file(out)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    assert agg["count"] == 50
    assert 0.0 <= agg["overall"] <= 1.0


def test_per_framework_report_renders(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), runner=CorrectFakeRunner())
    rf = load_result_file(out)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    md = render_per_framework(rf.framework, rf.model, agg, scored)
    assert "openai-agents" in md
    assert "math-001" in md
