import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from contract.validate import validate_file, cli

VALID_QUESTION = {
    "id": "math-001",
    "subject": "math",
    "type": "multiple_choice",
    "question": "Simplify: (x+2)(x-2)",
    "options": ["A. x^2-4", "B. x^2+4", "C. x^2-2x", "D. x^2-4x+4"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": {"max_points": 1, "method": "exact"},
    "source": "test",
    "difficulty": "easy",
}

VALID_RESULT = {
    "framework": "test",
    "model": "test-model",
    "generated_at": "2026-06-21T10:00:00Z",
    "config": {},
    "results": [
        {
            "question_id": "math-001",
            "subject": "math",
            "type": "multiple_choice",
            "response": "A",
            "reasoning": "",
            "latency_ms": 100,
            "raw": {},
        }
    ],
}


def _write(tmp_path: Path, obj: dict) -> Path:
    p = tmp_path / "data.json"
    p.write_text(json.dumps(obj))
    return p


def test_validate_question_ok(tmp_path):
    p = _write(tmp_path, VALID_QUESTION)
    assert validate_file(p, schema_name="question") is True


def test_validate_result_ok(tmp_path):
    p = _write(tmp_path, VALID_RESULT)
    assert validate_file(p, schema_name="result") is True


def test_validate_question_bad_subject(tmp_path):
    bad = dict(VALID_QUESTION, subject="art")
    p = _write(tmp_path, bad)
    with pytest.raises(Exception):
        validate_file(p, schema_name="question")


def test_validate_mc_requires_options(tmp_path):
    bad = dict(VALID_QUESTION)
    bad = {k: v for k, v in bad.items() if k != "options"}
    p = _write(tmp_path, bad)
    with pytest.raises(Exception):
        validate_file(p, schema_name="question")


def test_cli_exit_codes(tmp_path):
    good = _write(tmp_path, VALID_QUESTION)
    runner = CliRunner()
    res = runner.invoke(cli, ["--file", str(good), "--schema", "question"])
    assert res.exit_code == 0
