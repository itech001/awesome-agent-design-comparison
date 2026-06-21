import json
from collections import Counter
from pathlib import Path

import jsonschema

_DATASET = Path(__file__).resolve().parent.parent / "dataset" / "questions.json"
_SCHEMA = Path(__file__).resolve().parent.parent / "contract" / "question.schema.json"

EXPECTED_DIST = {
    # subject: (mc_count, short_count, total)
    "chinese": (5, 3, 8),
    "math": (4, 4, 8),
    "english": (5, 3, 8),
    "physics": (4, 2, 6),
    "chemistry": (4, 2, 6),
    "biology": (4, 2, 6),
    "history": (3, 1, 4),
    "geography": (3, 1, 4),
}


def _load():
    return json.loads(_DATASET.read_text())


def _schema():
    return json.loads(_SCHEMA.read_text())


def test_dataset_validates_against_schema():
    data = _load()
    assert isinstance(data, list)
    schema = _schema()
    for q in data:
        jsonschema.validate(q, schema)


def test_dataset_has_50_questions():
    assert len(_load()) == 50


def test_dataset_ids_unique():
    ids = [q["id"] for q in _load()]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    assert not dupes, f"duplicate ids: {dupes}"


def test_dataset_subject_distribution():
    counts = Counter((q["subject"], q["type"]) for q in _load())
    for subject, (mc, short, total) in EXPECTED_DIST.items():
        assert counts[(subject, "multiple_choice")] == mc, subject
        assert counts[(subject, "short_answer")] == short, subject


def test_dataset_rubric_questions_have_rubric():
    for q in _load():
        if q["scoring"]["method"] == "rubric":
            assert q["scoring"]["rubric"], q["id"]
            assert sum(r["points"] for r in q["scoring"]["rubric"]) == q["scoring"]["max_points"], q["id"]


def test_dataset_mc_has_options_and_letter_answer():
    for q in _load():
        if q["type"] == "multiple_choice":
            assert len(q["options"]) >= 2, q["id"]
            assert q["answer_type"] == "letter", q["id"]
            assert len(q["answer"]) == 1, q["id"]


def test_dataset_short_answer_is_text_and_rubric():
    for q in _load():
        if q["type"] == "short_answer":
            assert q["answer_type"] == "text", q["id"]
            assert q["scoring"]["method"] == "rubric", q["id"]
            assert "options" not in q, q["id"]
