import json
import sys
from pathlib import Path

import jsonschema

import run as run_module
from output_models import AgentAnswer

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_RESULT_SCHEMA = _REPO / "contract" / "result.schema.json"


class FakeRunner:
    def __init__(self):
        self.count = 0

    def __call__(self, agent, prompt):
        self.count += 1
        return AgentAnswer(response="A", reasoning="fake reasoning")


def test_run_writes_valid_result_file(tmp_path):
    out = tmp_path / "results.json"
    fake = FakeRunner()
    run_module.run(
        dataset_path=str(_DATASET),
        output_path=str(out),
        framework="openai-agents",
        model="gpt-4o",
        runner=fake,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    schema = json.loads(_RESULT_SCHEMA.read_text())
    jsonschema.validate(data, schema)
    assert data["framework"] == "openai-agents"
    assert data["model"] == "gpt-4o"
    assert len(data["results"]) == 50
    for r in data["results"]:
        assert set(r.keys()) >= {"question_id", "subject", "type", "response", "reasoning", "latency_ms", "raw"}


def test_run_preserves_subject_and_type(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(
        dataset_path=str(_DATASET),
        output_path=str(out),
        runner=FakeRunner(),
    )
    data = json.loads(out.read_text())
    by_id = {r["question_id"]: r for r in data["results"]}
    assert by_id["math-001"]["subject"] == "math"
    assert by_id["math-001"]["type"] == "multiple_choice"
    assert by_id["chinese-006"]["type"] == "short_answer"


def test_run_records_latency(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), runner=FakeRunner())
    data = json.loads(out.read_text())
    assert all(r["latency_ms"] >= 0 for r in data["results"])


def test_run_handles_runner_error_records_empty_response(tmp_path):
    out = tmp_path / "results.json"

    def bad_runner(agent, prompt):
        raise RuntimeError("API down")

    run_module.run(dataset_path=str(_DATASET), output_path=str(out), runner=bad_runner)
    data = json.loads(out.read_text())
    assert len(data["results"]) == 50
    for r in data["results"]:
        assert r["response"] == ""
        assert "error" in r["raw"]


def test_run_records_validator_status_when_revises(tmp_path):
    """When the inner loop revises, run.py records attempts/accepted/validator_answer."""
    from output_models import AgentAnswer, ValidatorVerdict

    out = tmp_path / "results.json"

    class ScriptedResolver:
        def __init__(self):
            self.i = 0

        def __call__(self, agent, prompt):
            self.i += 1
            # First attempt wrong, second attempt right.
            return AgentAnswer(response="B" if self.i == 1 else "A", reasoning="r")

    class ScriptedValidator:
        def __init__(self):
            self.i = 0

        def __call__(self, agent, prompt):
            self.i += 1
            if self.i == 1:
                return ValidatorVerdict(accepted=False, feedback="wrong", checked_answer="A")
            return ValidatorVerdict(accepted=True, checked_answer="A")

    run_module.run(
        dataset_path=str(_DATASET),
        output_path=str(out),
        resolver_runner=ScriptedResolver(),
        validator_runner=ScriptedValidator(),
    )
    data = json.loads(out.read_text())
    # math-001 answer is A; first MC question should have revised to A.
    first_mc = next(r for r in data["results"] if r["type"] == "multiple_choice")
    assert first_mc["response"] == "A"
    assert first_mc["raw"]["attempts"] == 2
    assert first_mc["raw"]["accepted"] is True
    assert first_mc["raw"]["validator_answer"] == "A"
