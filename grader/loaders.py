"""Load and validate dataset/result files into pydantic models."""
from __future__ import annotations

import json
from pathlib import Path

from grader.models import Question, ResultFile


def load_dataset(path: str | Path) -> list[Question]:
    """Load dataset/questions.json into a list of Question models."""
    data = json.loads(Path(path).read_text())
    return [Question(**q) for q in data]


def load_questions_by_id(path: str | Path) -> dict[str, Question]:
    """Load the dataset indexed by question id."""
    return {q.id: q for q in load_dataset(path)}


def load_result_file(path: str | Path) -> ResultFile:
    """Load a framework's result/results.json into a ResultFile model."""
    data = json.loads(Path(path).read_text())
    return ResultFile(**data)
