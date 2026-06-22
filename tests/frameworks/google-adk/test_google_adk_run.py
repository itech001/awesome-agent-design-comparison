import json
from pathlib import Path

import jsonschema

import run as run_module

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_RESULT_SCHEMA = _REPO / "contract" / "result.schema.json"


class FakeRunner:
    def __init__(self):
        self.count = 0

    def __call__(self, agent, prompt):
        self.count += 1
        return '{"response": "A", "reasoning": "fake"}'


def test_run_writes_valid_result_file(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(
        dataset_path=str(_DATASET),
        output_path=str(out),
        framework="google-adk",
        model="gemini-2.5-flash",
        runner=FakeRunner(),
    )
    assert out.exists()
    data = json.loads(out.read_text())
    schema = json.loads(_RESULT_SCHEMA.read_text())
    jsonschema.validate(data, schema)
    assert data["framework"] == "google-adk"
    assert data["model"] == "gemini-2.5-flash"
    assert len(data["results"]) == 50


def test_run_preserves_subject_and_type(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(
        dataset_path=str(_DATASET), output_path=str(out), runner=FakeRunner()
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
