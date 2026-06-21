import json
from pathlib import Path

import pytest

from grader.loaders import load_dataset, load_result_file, load_questions_by_id

_FIXTURE = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "sample_result.json"
_DATASET = Path(__file__).resolve().parent.parent.parent / "dataset" / "questions.json"


def test_load_dataset_returns_list_of_questions():
    qs = load_dataset(_DATASET)
    assert len(qs) == 50
    assert all(hasattr(q, "id") for q in qs)


def test_load_result_file_ok():
    rf = load_result_file(_FIXTURE)
    assert rf.framework == "sample"
    assert len(rf.results) == 2
    assert rf.results[0].response == "A"


def test_load_questions_by_id_keys_match():
    by_id = load_questions_by_id(_DATASET)
    assert set(by_id.keys()) == {q.id for q in load_dataset(_DATASET)}


def test_load_result_file_rejects_missing_required_field(tmp_path):
    bad = {"framework": "x", "model": "y"}  # missing generated_at, results
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(Exception):
        load_result_file(p)
