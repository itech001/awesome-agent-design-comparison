"""End-to-end run.py tests with a fake LLM (no API calls)."""
import json
from pathlib import Path

import jsonschema

import run as run_module

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_RESULT_SCHEMA = _REPO / "contract" / "result.schema.json"


class FakeLLM:
    """Always returns an accepted 'A' for any prompt."""

    def __init__(self):
        self.count = 0

    def invoke(self, prompt: str) -> str:
        self.count += 1
        if "verify" in prompt.lower():
            return '{"accepted": true, "feedback": "", "checked_answer": "A"}'
        return '{"response": "A", "reasoning": "fake"}'


def test_run_writes_valid_result_file(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(
        dataset_path=str(_DATASET),
        output_path=str(out),
        framework="langgraph",
        model="gpt-4o",
        llm=FakeLLM(),
    )
    assert out.exists()
    data = json.loads(out.read_text())
    schema = json.loads(_RESULT_SCHEMA.read_text())
    jsonschema.validate(data, schema)
    assert data["framework"] == "langgraph"
    assert data["model"] == "gpt-4o"
    assert len(data["results"]) == 50
    for r in data["results"]:
        assert set(r.keys()) >= {"question_id", "subject", "type", "response", "reasoning", "latency_ms", "raw"}


def test_run_preserves_subject_and_type(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), llm=FakeLLM())
    data = json.loads(out.read_text())
    by_id = {r["question_id"]: r for r in data["results"]}
    assert by_id["math-001"]["subject"] == "math"
    assert by_id["math-001"]["type"] == "multiple_choice"
    assert by_id["chinese-006"]["type"] == "short_answer"


def test_run_records_latency(tmp_path):
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), llm=FakeLLM())
    data = json.loads(out.read_text())
    assert all(r["latency_ms"] >= 0 for r in data["results"])


def test_run_handles_error_records_empty_response(tmp_path):
    out = tmp_path / "results.json"

    class BadLLM:
        def invoke(self, prompt: str) -> str:
            raise RuntimeError("API down")

    run_module.run(dataset_path=str(_DATASET), output_path=str(out), llm=BadLLM())
    data = json.loads(out.read_text())
    assert len(data["results"]) == 50
    for r in data["results"]:
        assert r["response"] == ""
        assert "error" in r["raw"]


def test_run_records_validator_status(tmp_path):
    """run.py records attempts/accepted/validator_answer from the graph."""
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), llm=FakeLLM())
    data = json.loads(out.read_text())
    first = data["results"][0]
    assert "attempts" in first["raw"]
    assert first["raw"]["accepted"] is True
    assert first["raw"]["validator_answer"] == "A"
