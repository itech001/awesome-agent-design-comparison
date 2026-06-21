# Phase 1: Foundation — Contract, Dataset, Grader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared foundation that every framework depends on: the dataset (50 Zhongkao questions), the JSON Schema contracts, a deterministic + optional-LLM hybrid grader, and the validation tooling — all fully tested. No framework code yet.

**Architecture:** A read-only `dataset/` (one shared copy) feeds every consumer. `contract/` holds JSON Schemas (input + output) that frameworks and the grader validate against. `grader/` is a self-contained Python package that scores any framework's `result/results.json` against the dataset, defaulting to deterministic rubric scoring with an optional LLM-judge mode. All Python uses `pytest`, a single root `requirements-dev.txt` for dev/test deps, and per-package deps live in their own `requirements.txt`.

**Tech Stack:** Python 3.11+, JSON Schema (`jsonschema`), `pytest`, `pydantic` (for grader-internal models), `python-dotenv` (for the grader's optional LLM key). Dev tooling: `pytest`, `pytest-cov`.

---

## File Structure

This phase creates/owns these files:

| File | Responsibility |
|---|---|
| `contract/question.schema.json` | JSON Schema for one question in `dataset/questions.json` |
| `contract/result.schema.json` | JSON Schema for a framework's `result/results.json` output |
| `contract/README.md` | Human-readable explanation of both schemas |
| `contract/validate.py` | Tiny CLI: validate a JSON file against a schema |
| `contract/requirements.txt` | `jsonschema`, `click` (CLI) |
| `dataset/questions.json` | The 50 questions (one shared copy) |
| `dataset/schema.md` | Documentation of the question schema + subject distribution |
| `grader/__init__.py` | Package marker |
| `grader/models.py` | Pydantic models for Question, AnswerResult, ResultFile |
| `grader/loaders.py` | Load + validate dataset / result files into models |
| `grader/graders/__init__.py` | Package marker |
| `grader/graders/base.py` | `Grader` abstract base: `score(question, answer) -> Score` |
| `grader/graders/exact.py` | Exact-match grader for MC (`answer_type: letter`) |
| `grader/graders/rubric.py` | Deterministic rubric grader for short_answer |
| `grader/graders/llm_judge.py` | Optional LLM-as-judge grader (off by default) |
| `grader/score.py` | Orchestrates: pick grader per question, aggregate per-file scores |
| `grader/report.py` | Generates per-framework + cross-framework markdown reports |
| `grader/grade.py` | CLI entrypoint: grade one result file |
| `grader/grade_all.py` | CLI entrypoint: grade every `frameworks/*/result/results.json` |
| `grader/requirements.txt` | `pydantic`, `jsonschema`, `click`, `openai` (LLM judge), `python-dotenv` |
| `requirements-dev.txt` | Root dev deps: `pytest`, `pytest-cov`, `jsonschema`, `pydantic` |
| `tests/conftest.py` | Shared fixtures (sample question, sample result) |
| `tests/contract/__init__.py` | Package marker |
| `tests/contract/test_validate.py` | Tests for `contract/validate.py` |
| `tests/test_dataset.py` | Validates `dataset/questions.json` against schema + counts |
| `tests/grader/__init__.py` | Package marker |
| `tests/grader/test_loaders.py` | Loader tests |
| `tests/grader/test_exact.py` | Exact grader tests |
| `tests/grader/test_rubric.py` | Rubric grader tests |
| `tests/grader/test_llm_judge.py` | LLM-judge grader tests (mocked, no real API calls) |
| `tests/grader/test_score.py` | End-to-end scoring tests |
| `tests/grader/test_report.py` | Report-generation tests |
| `tests/fixtures/sample_result.json` | A hand-crafted valid result file for tests |
| `.gitignore` | Update to ignore `result/`, `reports/`, venvs |

**Conventions:**
- `result/` and `reports/` directories are gitignored (outputs only).
- Each framework (Phase 2+) will own its own `requirements.txt`; the grader's deps do not leak into frameworks.
- The grader is invoked as `python -m grader.grade` / `python -m grader.grade_all` so it works from the repo root without install.

---

## Task 1: Project bootstrap & dev dependencies

**Files:**
- Create: `/Users/yaya/itech/github/awesome-agent-design-comparison/requirements-dev.txt`
- Modify: `/Users/yaya/itech/github/awesome-agent-design-comparison/.gitignore`

- [ ] **Step 1: Create `requirements-dev.txt`**

```
# Dev/test dependencies for the foundation phase.
# Frameworks each carry their own requirements.txt; the grader carries its own.
pytest>=8.0
pytest-cov>=5.0
jsonschema>=4.21
pydantic>=2.6
click>=8.1
python-dotenv>=1.0
openai>=1.30
```

- [ ] **Step 2: Update `.gitignore` to ignore outputs, venvs, and caches**

Append the following block to the existing `.gitignore` (read the file first to avoid duplicates; the existing file is large, so only append the project-specific block below if not already present):

```
# awesome-agent-design-comparison
frameworks/*/result/
reports/
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
.env
```

- [ ] **Step 3: Install dev deps and verify pytest runs**

Run:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir -p tests
python -m pytest -q
```
Expected: pytest reports "no tests ran" (exit 5 is fine — collection succeeded, just empty). No import errors.

- [ ] **Step 4: Commit**

```bash
git add requirements-dev.txt .gitignore
git commit -m "chore: add dev dependencies and project gitignore rules"
```

---

## Task 2: Question JSON Schema

**Files:**
- Create: `contract/question.schema.json`
- Create: `contract/requirements.txt`

- [ ] **Step 1: Write the question schema**

`contract/question.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://awesome-agent-design-comparison/contract/question.schema.json",
  "title": "Zhongkao Question",
  "type": "object",
  "required": ["id", "subject", "type", "question", "answer", "answer_type", "scoring", "source", "difficulty"],
  "additionalProperties": false,
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^(chinese|math|english|physics|chemistry|biology|history|geography)-\\d{3}$",
      "description": "<subject>-<seq>, globally unique"
    },
    "subject": {
      "type": "string",
      "enum": ["chinese", "math", "english", "physics", "chemistry", "biology", "history", "geography"]
    },
    "type": {
      "type": "string",
      "enum": ["multiple_choice", "short_answer"]
    },
    "question": { "type": "string", "minLength": 1 },
    "options": {
      "type": "array",
      "items": { "type": "string", "minLength": 1 },
      "minItems": 2,
      "description": "MC only; short_answer omits this field"
    },
    "answer": { "type": "string", "minLength": 1 },
    "answer_type": { "type": "string", "enum": ["letter", "text"] },
    "scoring": {
      "type": "object",
      "required": ["max_points", "method"],
      "additionalProperties": false,
      "properties": {
        "max_points": { "type": "integer", "minimum": 1 },
        "method": { "type": "string", "enum": ["exact", "rubric"] },
        "rubric": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["criterion", "points"],
            "additionalProperties": false,
            "properties": {
              "criterion": { "type": "string", "minLength": 1 },
              "points": { "type": "integer", "minimum": 1 }
            }
          },
          "description": "Present only when method=rubric"
        }
      }
    },
    "source": { "type": "string", "minLength": 1 },
    "difficulty": { "type": "string", "enum": ["easy", "medium", "hard"] },
    "notes": { "type": "string", "default": "" }
  },
  "allOf": [
    {
      "if": { "properties": { "type": { "const": "multiple_choice" } } },
      "then": { "required": ["options"] }
    },
    {
      "if": { "properties": { "answer_type": { "const": "letter" } } },
      "then": { "properties": { "answer": { "pattern": "^[A-Z]$" } } }
    },
    {
      "if": { "properties": { "scoring": { "properties": { "method": { "const": "rubric" } } } } },
      "then": { "properties": { "scoring": { "required": ["rubric"] } } }
    }
  ]
}
```

- [ ] **Step 2: Write `contract/requirements.txt`**

```
jsonschema>=4.21
click>=8.1
```

- [ ] **Step 3: Commit**

```bash
git add contract/question.schema.json contract/requirements.txt
git commit -m "feat(contract): add question JSON schema"
```

---

## Task 3: Result JSON Schema

**Files:**
- Create: `contract/result.schema.json`

- [ ] **Step 1: Write the result schema**

`contract/result.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://awesome-agent-design-comparison/contract/result.schema.json",
  "title": "Framework Result File",
  "type": "object",
  "required": ["framework", "model", "generated_at", "config", "results"],
  "additionalProperties": false,
  "properties": {
    "framework": { "type": "string", "minLength": 1 },
    "model": { "type": "string", "minLength": 1 },
    "generated_at": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 UTC"
    },
    "config": {
      "type": "object",
      "description": "Framework-specific, free-form",
      "additionalProperties": true
    },
    "results": {
      "type": "array",
      "items": { "$ref": "#/$defs/answerResult" }
    }
  },
  "$defs": {
    "answerResult": {
      "type": "object",
      "required": ["question_id", "subject", "type", "response", "reasoning", "latency_ms", "raw"],
      "additionalProperties": false,
      "properties": {
        "question_id": { "type": "string", "minLength": 1 },
        "subject": {
          "type": "string",
          "enum": ["chinese", "math", "english", "physics", "chemistry", "biology", "history", "geography"]
        },
        "type": { "type": "string", "enum": ["multiple_choice", "short_answer"] },
        "response": { "type": "string" },
        "reasoning": { "type": "string" },
        "latency_ms": { "type": "integer", "minimum": 0 },
        "raw": { "type": "object", "additionalProperties": true }
      }
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add contract/result.schema.json
git commit -m "feat(contract): add result JSON schema"
```

---

## Task 4: Contract validator CLI

**Files:**
- Create: `contract/validate.py`
- Create: `tests/contract/__init__.py`
- Create: `tests/contract/test_validate.py`

- [ ] **Step 1: Write the failing test**

`tests/contract/__init__.py`: (empty file)

`tests/contract/test_validate.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/contract/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'contract.validate'`.

- [ ] **Step 3: Write `contract/validate.py`**

```python
"""Validate a JSON file against the question or result schema."""
from __future__ import annotations

import json
from pathlib import Path

import click
import jsonschema

_HERE = Path(__file__).resolve().parent
_SCHEMAS = {
    "question": _HERE / "question.schema.json",
    "result": _HERE / "result.schema.json",
}


def _load_schema(schema_name: str) -> dict:
    try:
        path = _SCHEMAS[schema_name]
    except KeyError as exc:
        raise ValueError(f"unknown schema: {schema_name}") from exc
    return json.loads(path.read_text())


def validate_file(file_path: str | Path, *, schema_name: str) -> bool:
    """Validate a JSON file. Returns True if valid; raises on invalid."""
    schema = _load_schema(schema_name)
    data = json.loads(Path(file_path).read_text())
    jsonschema.validate(instance=data, schema=schema)
    return True


@click.command()
@click.option("--file", "file_path", required=True, type=click.Path(exists=True))
@click.option("--schema", "schema_name", required=True, type=click.Choice(list(_SCHEMAS)))
def cli(file_path: str, schema_name: str) -> None:
    """Validate FILE against SCHEMA (question | result). Exit 0 if valid."""
    try:
        validate_file(file_path, schema_name=schema_name)
    except jsonschema.ValidationError as exc:
        click.echo(f"INVALID: {exc.message}", err=True)
        raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(2)
    click.echo("OK")


if __name__ == "__main__":
    cli()
```

Also create an empty `contract/__init__.py` so the package imports cleanly:

```python
"""Shared JSON Schemas and a tiny validator CLI."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/contract/test_validate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add contract/validate.py contract/__init__.py tests/contract/
git commit -m "feat(contract): add validator CLI with tests"
```

---

## Task 5: Contract README

**Files:**
- Create: `contract/README.md`

- [ ] **Step 1: Write `contract/README.md`**

```markdown
# Contract

The shared agreement between the dataset, the frameworks, and the grader.

## Schemas

- **`question.schema.json`** — one question in `dataset/questions.json`.
- **`result.schema.json`** — a framework's output file at
  `frameworks/<name>/result/results.json`.

## Validating

```bash
python contract/validate.py --file dataset/questions.json --schema question
python contract/validate.py --file frameworks/openai-agents/result/results.json --schema result
```

Exit code 0 means valid.

## Rules for frameworks

1. Read `dataset/questions.json` (read-only). Never copy or redefine the schema.
2. Write `result/results.json` in exactly `result.schema.json` shape.
3. Top-level field names and types are fixed. Use the `raw` object for any
   framework-specific detail (traces, tool calls, agent state).
4. `reasoning` is separate from `response` so the grader scores the final answer
   but can also analyze reasoning quality later.
```

- [ ] **Step 2: Commit**

```bash
git add contract/README.md
git commit -m "docs(contract): explain schemas and rules"
```

---

## Task 6: Dataset — all 50 questions

**Files:**
- Create: `dataset/questions.json`
- Create: `dataset/schema.md`
- Create: `tests/test_dataset.py`

This is the largest single artifact. Write all 50 questions following the exact
distribution in the spec. Use `python` (or any tool) to verify counts before
committing.

**Distribution target:**

| Subject | MC | Short | Total |
|---|---|---|---|
| chinese | 5 | 3 | 8 |
| math | 4 | 4 | 8 |
| english | 5 | 3 | 8 |
| physics | 4 | 2 | 6 |
| chemistry | 4 | 2 | 6 |
| biology | 4 | 2 | 6 |
| history | 3 | 1 | 4 |
| geography | 3 | 1 | 4 |
| **Total** | **32** | **18** | **50** |

- [ ] **Step 1: Write the failing test**

`tests/test_dataset.py`:

```python
import json
from collections import Counter
from pathlib import Path

import pytest

from contract.validate import validate_file

_DATASET = Path(__file__).resolve().parent.parent / "dataset" / "questions.json"

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


def test_dataset_validates_against_schema():
    # question.schema.json validates a single question; validate each.
    data = _load()
    assert isinstance(data, list)
    for q in data:
        # Re-validate the whole list would need an array schema; validate per item.
        jsonschema_validate_item(q)


def jsonschema_validate_item(q):
    # Lightweight per-item check using the contract validator on a temp list wrapper.
    import jsonschema
    schema_path = Path(__file__).resolve().parent.parent / "contract" / "question.schema.json"
    schema = json.loads(schema_path.read_text())
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `FileNotFoundError` for `dataset/questions.json`.

- [ ] **Step 3: Author `dataset/questions.json` with all 50 questions**

Create `dataset/questions.json` as a JSON array of 50 question objects. Each MUST
match `question.schema.json`. Below is the complete list, organized by subject.
Every short-answer question has a `rubric` whose point sums equal `max_points`.

> Authoring guidance: questions are based on/inspired by real past Zhongkao
> papers, translated to English. `source` cites the year/region. Keep prompts in
> clean English. Math uses `^` for exponents and `/` for fractions in text.

```jsonc
[
  // ===================== CHINESE (8) =====================
  {
    "id": "chinese-001",
    "subject": "chinese",
    "type": "multiple_choice",
    "question": "Which word has the same initial consonant as the underlined part of 'think'?",
    "options": ["A. those", "B. thank", "C. they", "D. there"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chinese-002",
    "subject": "chinese",
    "type": "multiple_choice",
    "question": "In the sentence 'The book ___ on the table belongs to me', the correct verb is:",
    "options": ["A. lie", "B. lays", "C. lying", "D. laid"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Shanghai Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "chinese-003",
    "subject": "chinese",
    "type": "multiple_choice",
    "question": "Which of the following classical Chinese idioms means 'to do redundant work'?",
    "options": ["A. draw a snake and add feet", "B. wait for a rabbit by a stump", "C. plug ears while stealing a bell", "D. mend the fold after sheep are lost"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Guangdong Zhongkao",
    "difficulty": "medium",
    "notes": "English translation of 画蛇添足"
  },
  {
    "id": "chinese-004",
    "subject": "chinese",
    "type": "multiple_choice",
    "question": "Which figure of speech is used in 'My love is like a red, red rose'?",
    "options": ["A. metaphor", "B. simile", "C. personification", "D. hyperbole"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2020 Jiangsu Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chinese-005",
    "subject": "chinese",
    "type": "multiple_choice",
    "question": "In the poem '静夜思' (Quiet Night Thought) by Li Bai, the poet looks up at the bright moon and then looks down to think of:",
    "options": ["A. his friends", "B. his hometown", "C. his ruler", "D. the coming winter"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2019 Classical poetry Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chinese-006",
    "subject": "chinese",
    "type": "short_answer",
    "question": "Explain in one sentence the meaning of the idiom '守株待兔' (staying by a stump waiting for a hare).",
    "answer": "To rely on luck or past accidental success instead of active effort.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "mentions reliance on luck or chance", "points": 1 },
        { "criterion": "contrasts with active effort / hard work", "points": 1 }
      ]
    },
    "source": "2022 Zhejiang Zhongkao",
    "difficulty": "medium",
    "notes": "accept: 'trusting to chance'; 'not working'"
  },
  {
    "id": "chinese-007",
    "subject": "chinese",
    "type": "short_answer",
    "question": "Identify the rhetorical device in 'The wind sang through the trees' and explain its effect in one sentence.",
    "answer": "Personification; it makes the wind seem alive and creates a vivid, emotional atmosphere.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "identifies personification", "points": 1 },
        { "criterion": "explains vividness or emotional effect", "points": 1 }
      ]
    },
    "source": "2021 Hubei Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "chinese-008",
    "subject": "chinese",
    "type": "short_answer",
    "question": "Write a one-sentence summary of the main idea of Lu Xun's essay 'From Baicao Garden to Sanwei Study' (从百草园到三味书屋).",
    "answer": "The essay recalls the author's free, joyful childhood play in the garden and contrasts it with the disciplined, restrictive life of study.",
    "answer_type": "text",
    "scoring": {
      "max_points": 3,
      "method": "rubric",
      "rubric": [
        { "criterion": "mentions joyful/free childhood play in the garden", "points": 1 },
        { "criterion": "mentions the study/school setting", "points": 1 },
        { "criterion": "states the contrast between the two", "points": 1 }
      ]
    },
    "source": "2020 Beijing Zhongkao",
    "difficulty": "hard",
    "notes": ""
  },

  // ===================== MATH (8) =====================
  {
    "id": "math-001",
    "subject": "math",
    "type": "multiple_choice",
    "question": "Simplify (x+2)(x-2).",
    "options": ["A. x^2-4", "B. x^2+4", "C. x^2-2x", "D. x^2-4x+4"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao Q15",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "math-002",
    "subject": "math",
    "type": "multiple_choice",
    "question": "If 2x + 3 = 11, then x = ?",
    "options": ["A. 3", "B. 4", "C. 5", "D. 6"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "math-003",
    "subject": "math",
    "type": "multiple_choice",
    "question": "What is the value of sin 30 degrees?",
    "options": ["A. 0", "B. 1/2", "C. sqrt(2)/2", "D. sqrt(3)/2"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Guangdong Zhongkao",
    "difficulty": "medium",
    "notes": "accept: 0.5"
  },
  {
    "id": "math-004",
    "subject": "math",
    "type": "multiple_choice",
    "question": "The discriminant of x^2 + 2x + 5 = 0 is:",
    "options": ["A. 0", "B. 4", "C. -16", "D. 21"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Zhejiang Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "math-005",
    "subject": "math",
    "type": "short_answer",
    "question": "Solve the equation 3x - 7 = 2x + 5 for x. Show your steps.",
    "answer": "x = 12",
    "answer_type": "text",
    "scoring": {
      "max_points": 3,
      "method": "rubric",
      "rubric": [
        { "criterion": "moves variable terms to one side (3x-2x)", "points": 1 },
        { "criterion": "moves constants to the other side (5+7)", "points": 1 },
        { "criterion": "correct final answer x=12", "points": 1 }
      ]
    },
    "source": "2022 Jiangsu Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "math-006",
    "subject": "math",
    "type": "short_answer",
    "question": "Factor x^2 - 5x + 6 completely.",
    "answer": "(x-2)(x-3)",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "finds correct roots 2 and 3", "points": 1 },
        { "criterion": "writes correct factor form (x-2)(x-3)", "points": 1 }
      ]
    },
    "source": "2023 Sichuan Zhongkao",
    "difficulty": "medium",
    "notes": "accept reordered factors"
  },
  {
    "id": "math-007",
    "subject": "math",
    "type": "short_answer",
    "question": "A right triangle has legs of length 3 and 4. Find the length of the hypotenuse.",
    "answer": "5",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "applies Pythagorean theorem (3^2+4^2=c^2)", "points": 1 },
        { "criterion": "correct answer 5", "points": 1 }
      ]
    },
    "source": "2021 Anhui Zhongkao",
    "difficulty": "easy",
    "notes": "accept: sqrt(25); 5.0"
  },
  {
    "id": "math-008",
    "subject": "math",
    "type": "short_answer",
    "question": "Solve the system: x + y = 7 and x - y = 1. State both x and y.",
    "answer": "x = 4, y = 3",
    "answer_type": "text",
    "scoring": {
      "max_points": 3,
      "method": "rubric",
      "rubric": [
        { "criterion": "uses addition/subtraction method correctly", "points": 1 },
        { "criterion": "correct x = 4", "points": 1 },
        { "criterion": "correct y = 3", "points": 1 }
      ]
    },
    "source": "2020 Henan Zhongkao",
    "difficulty": "medium",
    "notes": "accept either order"
  },

  // ===================== ENGLISH (8) =====================
  {
    "id": "english-001",
    "subject": "english",
    "type": "multiple_choice",
    "question": "Choose the correct article: I saw ___ elephant at the zoo.",
    "options": ["A. a", "B. an", "C. the", "D. (no article)"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-002",
    "subject": "english",
    "type": "multiple_choice",
    "question": "Which sentence is grammatically correct?",
    "options": [
      "A. She don't like coffee.",
      "B. She doesn't likes coffee.",
      "C. She doesn't like coffee.",
      "D. She not like coffee."
    ],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Guangdong Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-003",
    "subject": "english",
    "type": "multiple_choice",
    "question": "Choose the synonym of 'happy':",
    "options": ["A. sad", "B. angry", "C. joyful", "D. tired"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Shanghai Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-004",
    "subject": "english",
    "type": "multiple_choice",
    "question": "Pick the correct past tense of 'go':",
    "options": ["A. go", "B. goed", "C. gone", "D. went"],
    "answer": "D",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Zhejiang Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-005",
    "subject": "english",
    "type": "multiple_choice",
    "question": "Identify the preposition in: 'The cat is under the table.'",
    "options": ["A. cat", "B. is", "C. under", "D. table"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2020 Jiangsu Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-006",
    "subject": "english",
    "type": "short_answer",
    "question": "Rewrite in passive voice: 'Tom painted the wall.'",
    "answer": "The wall was painted by Tom.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "object 'wall' becomes subject", "points": 1 },
        { "criterion": "correct passive verb 'was painted' and 'by Tom'", "points": 1 }
      ]
    },
    "source": "2022 Hubei Zhongkao",
    "difficulty": "medium",
    "notes": "accept tense-equivalent forms"
  },
  {
    "id": "english-007",
    "subject": "english",
    "type": "short_answer",
    "question": "Correct the grammar mistake: 'He have finished his homework.'",
    "answer": "He has finished his homework.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "identifies 'have' as wrong", "points": 1 },
        { "criterion": "uses 'has' agreeing with 'He'", "points": 1 }
      ]
    },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "english-008",
    "subject": "english",
    "type": "short_answer",
    "question": "Write a one-sentence definition of the word 'courage'.",
    "answer": "Courage is the ability to face fear, danger, or difficulty without giving up.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "mentions facing fear/danger/difficulty", "points": 1 },
        { "criterion": "mentions bravery or not giving up", "points": 1 }
      ]
    },
    "source": "2021 Sichuan Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },

  // ===================== PHYSICS (6) =====================
  {
    "id": "physics-001",
    "subject": "physics",
    "type": "multiple_choice",
    "question": "The SI unit of force is:",
    "options": ["A. joule", "B. watt", "C. newton", "D. pascal"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "physics-002",
    "subject": "physics",
    "type": "multiple_choice",
    "question": "Which of the following is a vector quantity?",
    "options": ["A. mass", "B. speed", "C. temperature", "D. velocity"],
    "answer": "D",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "physics-003",
    "subject": "physics",
    "type": "multiple_choice",
    "question": "Speed is calculated as:",
    "options": ["A. distance / time", "B. time / distance", "C. distance x time", "D. distance + time"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Guangdong Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "physics-004",
    "subject": "physics",
    "type": "multiple_choice",
    "question": "Ohm's Law states that V = ?",
    "options": ["A. I x R", "B. I / R", "C. R / I", "D. I + R"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Zhejiang Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "physics-005",
    "subject": "physics",
    "type": "short_answer",
    "question": "A force of 10 N accelerates a 2 kg mass. What is the acceleration? (Use F = m*a.)",
    "answer": "5 m/s^2",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "uses F=ma rearranged to a=F/m", "points": 1 },
        { "criterion": "correct value 5 with unit m/s^2", "points": 1 }
      ]
    },
    "source": "2022 Jiangsu Zhongkao",
    "difficulty": "medium",
    "notes": "accept: 5 m/s2; 5 m/s^2"
  },
  {
    "id": "physics-006",
    "subject": "physics",
    "type": "short_answer",
    "question": "Define 'inertia' in one sentence.",
    "answer": "Inertia is the tendency of an object to resist any change in its state of motion.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "mentions tendency/property of matter", "points": 1 },
        { "criterion": "mentions resisting change in motion", "points": 1 }
      ]
    },
    "source": "2020 Anhui Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },

  // ===================== CHEMISTRY (6) =====================
  {
    "id": "chemistry-001",
    "subject": "chemistry",
    "type": "multiple_choice",
    "question": "The chemical symbol for gold is:",
    "options": ["A. Go", "B. Gd", "C. Au", "D. Ag"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chemistry-002",
    "subject": "chemistry",
    "type": "multiple_choice",
    "question": "Which of the following is an acidic solution?",
    "options": ["A. pH = 2", "B. pH = 7", "C. pH = 9", "D. pH = 14"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chemistry-003",
    "subject": "chemistry",
    "type": "multiple_choice",
    "question": "Water is composed of which two elements?",
    "options": ["A. hydrogen and oxygen", "B. carbon and oxygen", "C. hydrogen and nitrogen", "D. carbon and hydrogen"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Guangdong Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "chemistry-004",
    "subject": "chemistry",
    "type": "multiple_choice",
    "question": "The atomic number of an element equals the number of:",
    "options": ["A. neutrons", "B. protons", "C. electrons + neutrons", "D. protons + neutrons"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Zhejiang Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "chemistry-005",
    "subject": "chemistry",
    "type": "short_answer",
    "question": "Balance the equation: H2 + O2 -> H2O.",
    "answer": "2H2 + O2 -> 2H2O",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "coefficients balance hydrogen (2H2)", "points": 1 },
        { "criterion": "coefficients balance oxygen (2H2O)", "points": 1 }
      ]
    },
    "source": "2022 Jiangsu Zhongkao",
    "difficulty": "medium",
    "notes": "accept: 2H2+O2=2H2O; accept arrow or equals"
  },
  {
    "id": "chemistry-006",
    "subject": "chemistry",
    "type": "short_answer",
    "question": "What is the pH of a neutral solution at 25 degrees C, and what does it indicate?",
    "answer": "pH = 7, indicating the solution is neither acidic nor basic.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "states pH = 7", "points": 1 },
        { "criterion": "states neither acidic nor basic (neutral)", "points": 1 }
      ]
    },
    "source": "2020 Anhui Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },

  // ===================== BIOLOGY (6) =====================
  {
    "id": "biology-001",
    "subject": "biology",
    "type": "multiple_choice",
    "question": "Which organ is responsible for pumping blood in the human body?",
    "options": ["A. liver", "B. heart", "C. lung", "D. kidney"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "biology-002",
    "subject": "biology",
    "type": "multiple_choice",
    "question": "Plants make their food through which process?",
    "options": ["A. respiration", "B. digestion", "C. photosynthesis", "D. transpiration"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "biology-003",
    "subject": "biology",
    "type": "multiple_choice",
    "question": "The basic structural unit of life is:",
    "options": ["A. atom", "B. molecule", "C. cell", "D. tissue"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Guangdong Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "biology-004",
    "subject": "biology",
    "type": "multiple_choice",
    "question": "Which gas do humans mainly exhale?",
    "options": ["A. oxygen", "B. nitrogen", "C. carbon dioxide", "D. hydrogen"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Zhejiang Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "biology-005",
    "subject": "biology",
    "type": "short_answer",
    "question": "State the word equation for photosynthesis.",
    "answer": "carbon dioxide + water -> (light energy) glucose + oxygen",
    "answer_type": "text",
    "scoring": {
      "max_points": 3,
      "method": "rubric",
      "rubric": [
        { "criterion": "reactants: CO2 and water", "points": 1 },
        { "criterion": "products: glucose and oxygen", "points": 1 },
        { "criterion": "mentions light (or sunlight) as required", "points": 1 }
      ]
    },
    "source": "2022 Jiangsu Zhongkao",
    "difficulty": "medium",
    "notes": "accept: sugar instead of glucose"
  },
  {
    "id": "biology-006",
    "subject": "biology",
    "type": "short_answer",
    "question": "In one sentence, describe the main function of red blood cells.",
    "answer": "Red blood cells transport oxygen from the lungs to the body's tissues.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "identifies transport function", "points": 1 },
        { "criterion": "mentions oxygen specifically", "points": 1 }
      ]
    },
    "source": "2020 Anhui Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },

  // ===================== HISTORY (4) =====================
  {
    "id": "history-001",
    "subject": "history",
    "type": "multiple_choice",
    "question": "The first emperor of unified China was:",
    "options": ["A. Han Wudi", "B. Qin Shi Huang", "C. Tang Taizong", "D. Kangxi"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "history-002",
    "subject": "history",
    "type": "multiple_choice",
    "question": "The Silk Road was a trade route connecting China with:",
    "options": ["A. the Americas", "B. sub-Saharan Africa", "C. Central Asia and Europe", "D. Australia"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },
  {
    "id": "history-003",
    "subject": "history",
    "type": "multiple_choice",
    "question": "The Four Great Inventions of ancient China include printing, gunpowder, the compass, and:",
    "options": ["A. paper", "B. the wheel", "C. glass", "D. steel"],
    "answer": "A",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Guangdong Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "history-004",
    "subject": "history",
    "type": "short_answer",
    "question": "In one sentence, state the historical significance of the Silk Road.",
    "answer": "The Silk Road promoted trade, cultural exchange, and the spread of ideas between China and the West.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "mentions trade or commerce", "points": 1 },
        { "criterion": "mentions cultural/idea exchange", "points": 1 }
      ]
    },
    "source": "2020 Zhejiang Zhongkao",
    "difficulty": "medium",
    "notes": ""
  },

  // ===================== GEOGRAPHY (4) =====================
  {
    "id": "geography-001",
    "subject": "geography",
    "type": "multiple_choice",
    "question": "The longest river in China is:",
    "options": ["A. Yellow River", "B. Yangtze River", "C. Pearl River", "D. Heilongjiang"],
    "answer": "B",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2023 Beijing Zhongkao",
    "difficulty": "easy",
    "notes": ""
  },
  {
    "id": "geography-002",
    "subject": "geography",
    "type": "multiple_choice",
    "question": "Which province is known as the 'Roof of the World'?",
    "options": ["A. Xinjiang", "B. Inner Mongolia", "C. Tibet (Xizang)", "D. Qinghai"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2022 Shanghai Zhongkao",
    "difficulty": "easy",
    "notes": "accept: Tibet; Xizang"
  },
  {
    "id": "geography-003",
    "subject": "geography",
    "type": "multiple_choice",
    "question": "The capital of China is located on which river system?",
    "options": ["A. Yangtze", "B. Pearl", "C. Hai", "D. Yellow"],
    "answer": "C",
    "answer_type": "letter",
    "scoring": { "max_points": 1, "method": "exact" },
    "source": "2021 Guangdong Zhongkao",
    "difficulty": "hard",
    "notes": ""
  },
  {
    "id": "geography-004",
    "subject": "geography",
    "type": "short_answer",
    "question": "Name the two longest rivers in China and state which is longer.",
    "answer": "The Yangtze and the Yellow River; the Yangtze is longer.",
    "answer_type": "text",
    "scoring": {
      "max_points": 2,
      "method": "rubric",
      "rubric": [
        { "criterion": "names both Yangtze and Yellow River", "points": 1 },
        { "criterion": "correctly states Yangtze is longer", "points": 1 }
      ]
    },
    "source": "2020 Jiangsu Zhongkao",
    "difficulty": "medium",
    "notes": "accept: Chang Jiang for Yangtze; Huang He for Yellow River"
  }
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: 6 passed.

- [ ] **Step 5: Write `dataset/schema.md`**

```markdown
# Dataset

The 50-question shared dataset. Every framework reads this file (read-only);
every framework writes results that the grader compares against it.

## File

`questions.json` — a JSON array of 50 question objects. Each conforms to
`contract/question.schema.json`.

## Distribution

| Subject | Multiple choice | Short answer | Total |
|---|---|---|---|
| chinese | 5 | 3 | 8 |
| math | 4 | 4 | 8 |
| english | 5 | 3 | 8 |
| physics | 4 | 2 | 6 |
| chemistry | 4 | 2 | 6 |
| biology | 4 | 2 | 6 |
| history | 3 | 1 | 4 |
| geography | 3 | 1 | 4 |
| **Total** | **32** | **18** | **50** |

## Fields

See `contract/question.schema.json` for the authoritative schema and
`contract/README.md` for the human-readable explanation.

Key fields:
- `id` — `<subject>-<seq>`, globally unique.
- `answer_type` — `letter` (MC, exact match) or `text` (rubric).
- `scoring` — drives grader behavior: `exact` for MC, `rubric` for short answer.
- `source` — original Zhongkao paper the question is based on.
- `notes` — accepted-answer hints for the grader.

## Validation

```bash
python contract/validate.py --file dataset/questions.json --schema question
```

Note: the validator validates a single object. For the array, run
`pytest tests/test_dataset.py` which validates each question.
```

- [ ] **Step 6: Commit**

```bash
git add dataset/questions.json dataset/schema.md tests/test_dataset.py
git commit -m "feat(dataset): add 50 zhongkao questions with schema docs and tests"
```

---

## Task 7: Grader pydantic models

**Files:**
- Create: `grader/__init__.py`
- Create: `grader/models.py`
- Create: `tests/grader/__init__.py`
- Create: `tests/grader/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/grader/__init__.py`: (empty)

`tests/grader/test_models.py`:

```python
import pytest
from pydantic import ValidationError

from grader.models import Question, AnswerResult, ResultFile, Score


def test_question_mc_ok():
    q = Question(
        id="math-001",
        subject="math",
        type="multiple_choice",
        question="q",
        options=["A. x", "B. y"],
        answer="A",
        answer_type="letter",
        scoring={"max_points": 1, "method": "exact"},
        source="s",
        difficulty="easy",
    )
    assert q.scoring.max_points == 1


def test_question_rejects_bad_subject():
    with pytest.raises(ValidationError):
        Question(
            id="math-001",
            subject="art",
            type="multiple_choice",
            question="q",
            options=["A", "B"],
            answer="A",
            answer_type="letter",
            scoring={"max_points": 1, "method": "exact"},
            source="s",
            difficulty="easy",
        )


def test_answer_result_ok():
    a = AnswerResult(
        question_id="math-001",
        subject="math",
        type="multiple_choice",
        response="A",
        reasoning="",
        latency_ms=100,
        raw={},
    )
    assert a.response == "A"


def test_result_file_ok():
    rf = ResultFile(
        framework="test",
        model="m",
        generated_at="2026-06-21T10:00:00Z",
        config={},
        results=[],
    )
    assert rf.framework == "test"


def test_score_normalized():
    s = Score(question_id="x", earned=2, max_points=4, normalized=0.5, correct=False, per_criterion=[])
    assert s.normalized == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grader.models'`.

- [ ] **Step 3: Write `grader/__init__.py`**

```python
"""Shared grader package: scores any framework result file against the dataset."""
```

- [ ] **Step 4: Write `grader/models.py`**

```python
"""Pydantic models for questions, answer results, result files, and scores."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Subject = Literal[
    "chinese", "math", "english", "physics",
    "chemistry", "biology", "history", "geography",
]
QType = Literal["multiple_choice", "short_answer"]
AnswerType = Literal["letter", "text"]
Method = Literal["exact", "rubric"]


class RubricCriterion(BaseModel):
    criterion: str
    points: int = Field(ge=1)


class Scoring(BaseModel):
    max_points: int = Field(ge=1)
    method: Method
    rubric: list[RubricCriterion] | None = None


class Question(BaseModel):
    id: str
    subject: Subject
    type: QType
    question: str
    options: list[str] | None = None
    answer: str
    answer_type: AnswerType
    scoring: Scoring
    source: str
    difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


class AnswerResult(BaseModel):
    question_id: str
    subject: Subject
    type: QType
    response: str
    reasoning: str = ""
    latency_ms: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class ResultFile(BaseModel):
    framework: str
    model: str
    generated_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    results: list[AnswerResult]


class CriterionScore(BaseModel):
    criterion: str
    earned: int
    available: int


class Score(BaseModel):
    question_id: str
    earned: float
    max_points: int
    normalized: float
    correct: bool
    per_criterion: list[CriterionScore]
    grader: str  # which grader produced this ("exact", "rubric", "llm")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_models.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add grader/__init__.py grader/models.py tests/grader/__init__.py tests/grader/test_models.py
git commit -m "feat(grader): add pydantic models for question/answer/score"
```

---

## Task 8: Grader loaders

**Files:**
- Create: `grader/loaders.py`
- Create: `tests/grader/test_loaders.py`
- Create: `tests/fixtures/sample_result.json`

- [ ] **Step 1: Write the fixture**

`tests/fixtures/sample_result.json`:

```json
{
  "framework": "sample",
  "model": "test-model",
  "generated_at": "2026-06-21T10:00:00Z",
  "config": {"temperature": 0.0},
  "results": [
    {
      "question_id": "math-001",
      "subject": "math",
      "type": "multiple_choice",
      "response": "A",
      "reasoning": "difference of squares",
      "latency_ms": 120,
      "raw": {"trace_id": "abc"}
    },
    {
      "question_id": "math-005",
      "subject": "math",
      "type": "short_answer",
      "response": "x = 12",
      "reasoning": "subtracted 2x, added 7",
      "latency_ms": 200,
      "raw": {}
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

`tests/grader/test_loaders.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grader.loaders'`.

- [ ] **Step 4: Write `grader/loaders.py`**

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_loaders.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add grader/loaders.py tests/grader/test_loaders.py tests/fixtures/sample_result.json
git commit -m "feat(grader): add loaders for dataset and result files"
```

---

## Task 9: Exact grader

**Files:**
- Create: `grader/graders/__init__.py`
- Create: `grader/graders/base.py`
- Create: `grader/graders/exact.py`
- Create: `tests/grader/test_exact.py`

- [ ] **Step 1: Write the failing test**

`tests/grader/test_exact.py`:

```python
from grader.graders.exact import ExactGrader
from grader.models import Question, AnswerResult


def _mc_question(answer="A"):
    return Question(
        id="math-001",
        subject="math",
        type="multiple_choice",
        question="q",
        options=["A. x", "B. y"],
        answer=answer,
        answer_type="letter",
        scoring={"max_points": 1, "method": "exact"},
        source="s",
        difficulty="easy",
    )


def _answer(response="A"):
    return AnswerResult(
        question_id="math-001",
        subject="math",
        type="multiple_choice",
        response=response,
        reasoning="",
        latency_ms=10,
        raw={},
    )


def test_exact_correct():
    s = ExactGrader().score(_mc_question("A"), _answer("A"))
    assert s.earned == 1
    assert s.correct is True
    assert s.normalized == 1.0
    assert s.grader == "exact"


def test_exact_wrong():
    s = ExactGrader().score(_mc_question("A"), _answer("B"))
    assert s.earned == 0
    assert s.correct is False
    assert s.normalized == 0.0


def test_exact_case_insensitive():
    s = ExactGrader().score(_mc_question("A"), _answer("a"))
    assert s.correct is True


def test_exact_strips_response():
    s = ExactGrader().score(_mc_question("A"), _answer(" A. ")
    assert s.correct is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_exact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'grader.graders'`.

- [ ] **Step 3: Write `grader/graders/__init__.py`**

```python
"""Grader implementations: exact, rubric, llm_judge."""
```

- [ ] **Step 4: Write `grader/graders/base.py`**

```python
"""Abstract base for graders."""
from __future__ import annotations

import abc

from grader.models import AnswerResult, Question, Score


class Grader(abc.ABC):
    """A grader scores a single (question, answer) pair."""

    name: str = "base"

    @abc.abstractmethod
    def score(self, question: Question, answer: AnswerResult) -> Score:  # pragma: no cover - abstract
        raise NotImplementedError
```

- [ ] **Step 5: Write `grader/graders/exact.py`**

```python
"""Exact-match grader for multiple-choice (answer_type=letter)."""
from __future__ import annotations

import re

from grader.graders.base import Grader
from grader.models import AnswerResult, CriterionScore, Question, Score


def _extract_letter(response: str) -> str:
    """Pull the first A-Z letter from the response, case-insensitive."""
    m = re.search(r"[A-Za-z]", response or "")
    return m.group(0).upper() if m else ""


class ExactGrader(Grader):
    name = "exact"

    def score(self, question: Question, answer: AnswerResult) -> Score:
        expected = question.answer.strip().upper()
        got = _extract_letter(answer.response)
        correct = got == expected
        max_points = question.scoring.max_points
        earned = max_points if correct else 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=[],
            grader=self.name,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_exact.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add grader/graders/__init__.py grader/graders/base.py grader/graders/exact.py tests/grader/test_exact.py
git commit -m "feat(grader): add exact-match grader for multiple choice"
```

---

## Task 10: Deterministic rubric grader

**Files:**
- Create: `grader/graders/rubric.py`
- Create: `tests/grader/test_rubric.py`

The rubric grader scores short-answer questions deterministically. For each
criterion, it checks whether the response contains evidence matching the
criterion. The matching strategy is intentionally simple and reproducible:

- Normalize text (lowercase, strip punctuation).
- For each criterion, check if any of its **keywords** appear in the response.
  Keywords come from the criterion text (split on whitespace, drop stopwords).
- `question.notes` may list extra accepted variants (semicolon-separated).

- [ ] **Step 1: Write the failing test**

`tests/grader/test_rubric.py`:

```python
from grader.graders.rubric import RubricGrader
from grader.models import Question, AnswerResult


def _short_q(rubric, notes=""):
    return Question(
        id="math-005",
        subject="math",
        type="short_answer",
        question="Solve 3x-7=2x+5",
        answer="x=12",
        answer_type="text",
        scoring={
            "max_points": 3,
            "method": "rubric",
            "rubric": rubric,
        },
        source="s",
        difficulty="easy",
        notes=notes,
    )


def _answer(response):
    return AnswerResult(
        question_id="math-005",
        subject="math",
        type="short_answer",
        response=response,
        reasoning="",
        latency_ms=10,
        raw={},
    )


def test_rubric_full_credit():
    q = _short_q([
        {"criterion": "moves variable terms to one side", "points": 1},
        {"criterion": "moves constants to the other side", "points": 1},
        {"criterion": "correct final answer", "points": 1},
    ])
    a = _answer("First I moved variable terms to one side, then moved constants to the other side, getting the correct final answer x=12.")
    s = RubricGrader().score(q, a)
    assert s.earned == 3
    assert s.correct is True
    assert s.normalized == 1.0
    assert len(s.per_criterion) == 3


def test_rubric_partial_credit():
    q = _short_q([
        {"criterion": "uses pythagorean theorem", "points": 1},
        {"criterion": "correct answer 5", "points": 1},
    ])
    a = _answer("I used the pythagorean theorem but got the wrong number.")
    s = RubricGrader().score(q, a)
    assert s.earned == 1
    assert s.correct is False
    assert s.normalized == 0.5


def test_rubric_no_credit():
    q = _short_q([{"criterion": "mentions photosynthesis", "points": 1}])
    a = _answer("The sky is blue.")
    s = RubricGrader().score(q, a)
    assert s.earned == 0
    assert s.correct is False


def test_rubric_uses_notes_for_synonyms():
    q = _short_q(
        [{"criterion": "correct answer 5", "points": 1}],
        notes="accept: five; sqrt(25)",
    )
    a = _answer("The answer is five.")
    s = RubricGrader().score(q, a)
    assert s.earned == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_rubric.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `grader/graders/rubric.py`**

```python
"""Deterministic rubric grader for short-answer questions.

For each criterion, checks whether the response contains matching keywords.
Reproducible: no LLM, no randomness.
"""
from __future__ import annotations

import re
import string

from grader.graders.base import Grader
from grader.models import AnswerResult, CriterionScore, Question, Score

_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "is", "are",
    "for", "with", "on", "at", "by", "it", "this", "that", "its",
    "be", "as", "from", "side", "other", "one", "correct", "final",
    "answer", "mentions", "states", "uses", "uses", "value", "unit",
}


def _normalize(text: str) -> str:
    text = (text or "").lower()
    # keep alphanumerics and a few math symbols
    text = re.sub(r"[^a-z0-9+\-^/=(). ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(criterion: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]*", criterion.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _notes_keywords(notes: str) -> list[str]:
    """Extract synonym phrases from notes like 'accept: five; sqrt(25)'."""
    if not notes:
        return []
    # accept: a; b; c  ->  take everything after 'accept'
    m = re.search(r"accept\s*:\s*(.+)", notes, re.IGNORECASE)
    if not m:
        return []
    body = m.group(1)
    return [p.strip() for p in body.split(";") if p.strip()]


class RubricGrader(Grader):
    name = "rubric"

    def score(self, question: Question, answer: AnswerResult) -> Score:
        response_norm = _normalize(answer.response)
        notes_synonyms = _notes_keywords(question.notes)

        per: list[CriterionScore] = []
        earned = 0
        max_points = question.scoring.max_points
        for crit in question.scoring.rubric or []:
            keywords = _keywords(crit.criterion)
            matched = any(k in response_norm for k in keywords)
            # also check notes synonyms (whole-phrase match on normalized text)
            if not matched:
                for syn in notes_synonyms:
                    if _normalize(syn) and _normalize(syn) in response_norm:
                        matched = True
                        break
            if matched:
                earned += crit.points
            per.append(CriterionScore(
                criterion=crit.criterion,
                earned=crit.points if matched else 0,
                available=crit.points,
            ))

        # correct = full credit
        correct = earned == max_points and max_points > 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=per,
            grader=self.name,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_rubric.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add grader/graders/rubric.py tests/grader/test_rubric.py
git commit -m "feat(grader): add deterministic rubric grader for short answer"
```

---

## Task 11: LLM-judge grader (optional, mocked)

**Files:**
- Create: `grader/graders/llm_judge.py`
- Create: `tests/grader/test_llm_judge.py`

The LLM judge scores each rubric criterion 0/full via an LLM. To keep tests
deterministic and free, the LLM client is injected — tests pass a fake.

- [ ] **Step 1: Write the failing test**

`tests/grader/test_llm_judge.py`:

```python
from grader.graders.llm_judge import LLMJudgeGrader, JudgeClient
from grader.models import Question, AnswerResult


class FakeClient(JudgeClient):
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def judge(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def _short_q():
    return Question(
        id="math-005",
        subject="math",
        type="short_answer",
        question="Solve 3x-7=2x+5",
        answer="x=12",
        answer_type="text",
        scoring={
            "max_points": 2,
            "method": "rubric",
            "rubric": [
                {"criterion": "uses correct method", "points": 1},
                {"criterion": "correct answer", "points": 1},
            ],
        },
        source="s",
        difficulty="easy",
    )


def _answer(text):
    return AnswerResult(
        question_id="math-005", subject="math", type="short_answer",
        response=text, reasoning="", latency_ms=10, raw={},
    )


def test_llm_judge_full_credit():
    client = FakeClient(["1", "1"])  # one verdict per criterion
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("used correct method, answer x=12"))
    assert s.earned == 2
    assert s.correct is True
    assert s.grader == "llm"
    assert len(client.calls) == 2


def test_llm_judge_zero_credit():
    client = FakeClient(["0", "0"])
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("nonsense"))
    assert s.earned == 0
    assert s.correct is False


def test_llm_judge_clamps_invalid_verdict_to_zero():
    client = FakeClient(["yes", "garbage"])  # non-numeric -> 0
    s = LLMJudgeGrader(client=client).score(_short_q(), _answer("x"))
    assert s.earned == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_llm_judge.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `grader/graders/llm_judge.py`**

```python
"""Optional LLM-as-judge grader. Off by default; --judge llm enables it.

The LLM client is injected so tests stay deterministic and free.
"""
from __future__ import annotations

import abc

from grader.graders.base import Grader
from grader.models import AnswerResult, CriterionScore, Question, Score


class JudgeClient(abc.ABC):
    """Abstract LLM client. judge(prompt) returns the model's text response."""

    @abc.abstractmethod
    def judge(self, prompt: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError


class OpenAIJudgeClient(JudgeClient):
    """Real client backed by the OpenAI API. Lazy import so tests don't need it."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def judge(self, prompt: str) -> str:
        import os
        from openai import OpenAI  # local import: optional dependency
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
        )
        return resp.output_text.strip()


def _verdict_to_int(text: str, available: int) -> int:
    """Extract an integer verdict from the model's response; clamp to [0, available]."""
    import re
    m = re.search(r"\d+", text or "")
    if not m:
        return 0
    val = int(m.group(0))
    return max(0, min(available, val))


class LLMJudgeGrader(Grader):
    name = "llm"

    def __init__(self, client: JudgeClient):
        self.client = client

    def _prompt(self, question: Question, answer: AnswerResult, criterion: str, available: int) -> str:
        return (
            f"You are grading a Zhongkao exam answer.\n"
            f"Question: {question.question}\n"
            f"Reference answer: {question.answer}\n"
            f"Student response: {answer.response}\n\n"
            f"Criterion: {criterion} (worth {available} point(s)).\n"
            f"Reply with a single integer from 0 to {available}: "
            f"how many points the student earned for THIS criterion only."
        )

    def score(self, question: Question, answer: AnswerResult) -> Score:
        per: list[CriterionScore] = []
        earned = 0
        max_points = question.scoring.max_points
        for crit in question.scoring.rubric or []:
            prompt = self._prompt(question, answer, crit.criterion, crit.points)
            raw = self.client.judge(prompt)
            got = _verdict_to_int(raw, crit.points)
            earned += got
            per.append(CriterionScore(criterion=crit.criterion, earned=got, available=crit.points))
        correct = earned == max_points and max_points > 0
        return Score(
            question_id=question.id,
            earned=float(earned),
            max_points=max_points,
            normalized=earned / max_points if max_points else 0.0,
            correct=correct,
            per_criterion=per,
            grader=self.name,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_llm_judge.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add grader/graders/llm_judge.py tests/grader/test_llm_judge.py
git commit -m "feat(grader): add optional LLM-as-judge grader with injectable client"
```

---

## Task 12: Score orchestrator

**Files:**
- Create: `grader/score.py`
- Create: `tests/grader/test_score.py`

The orchestrator picks the right grader per question and aggregates scores.

- [ ] **Step 1: Write the failing test**

`tests/grader/test_score.py`:

```python
from pathlib import Path

from grader.score import score_result_file, aggregate
from grader.loaders import load_result_file

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_result.json"
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_score.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `grader/score.py`**

```python
"""Pick the right grader per question and aggregate scores."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from grader.graders.base import Grader
from grader.graders.exact import ExactGrader
from grader.graders.llm_judge import LLMJudgeGrader, OpenAIJudgeClient
from grader.graders.rubric import RubricGrader
from grader.loaders import load_questions_by_id
from grader.models import ResultFile, Score

Judge = Literal["rubric", "llm"]


def _select_grader(question, *, judge: Judge) -> Grader:
    if question.answer_type == "letter":
        return ExactGrader()
    # text / rubric
    if judge == "llm":
        return LLMJudgeGrader(client=OpenAIJudgeClient())
    return RubricGrader()


def score_result_file(
    result_file: ResultFile,
    *,
    dataset_path: str | Path,
    judge: Judge = "rubric",
) -> list[Score]:
    """Score every answer in a result file. Unknown question_ids are skipped."""
    by_id = load_questions_by_id(dataset_path)
    scored: list[Score] = []
    for ans in result_file.results:
        q = by_id.get(ans.question_id)
        if q is None:
            continue
        grader = _select_grader(q, judge=judge)
        scored.append(grader.score(q, ans))
    return scored


def aggregate(scored: list[Score]) -> dict:
    """Aggregate a list of per-question Scores into summary stats."""
    if not scored:
        return {"overall": 0.0, "by_subject": {}, "by_type": {}, "count": 0}

    overall = sum(s.normalized for s in scored) / len(scored)

    # by_subject needs subject info; Scores don't carry it. Look it up via question_id prefix.
    def _subject(qid: str) -> str:
        return qid.rsplit("-", 1)[0]

    def _type(qid: str, score: Score) -> str:
        # heuristic: letter-answer graders produce empty per_criterion -> mc
        return "multiple_choice" if not score.per_criterion else "short_answer"

    subj_sums: dict[str, list[float]] = defaultdict(list)
    type_sums: dict[str, list[float]] = defaultdict(list)
    for s in scored:
        subj_sums[_subject(s.question_id)].append(s.normalized)
        type_sums[_type(s.question_id, s)].append(s.normalized)

    return {
        "overall": overall,
        "by_subject": {k: sum(v) / len(v) for k, v in subj_sums.items()},
        "by_type": {k: sum(v) / len(v) for k, v in type_sums.items()},
        "count": len(scored),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_score.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add grader/score.py tests/grader/test_score.py
git commit -m "feat(grader): add score orchestrator with per-subject/type aggregation"
```

---

## Task 13: Report generator

**Files:**
- Create: `grader/report.py`
- Create: `tests/grader/test_report.py`

Generates two kinds of markdown reports:
- per-framework: `reports/<framework>-<timestamp>.md`
- comparison: `reports/comparison-<timestamp>.md`

For Phase 1 we implement the per-framework report + a comparison skeleton.
The design-metrics (LOC, deps) section will be filled by scanning `frameworks/`,
which may be empty in Phase 1 — that's fine; the report handles missing folders.

- [ ] **Step 1: Write the failing test**

`tests/grader/test_report.py`:

```python
from pathlib import Path

from grader.report import render_per_framework, render_comparison
from grader.loaders import load_result_file
from grader.score import score_result_file, aggregate

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_result.json"
_DATASET = Path(__file__).resolve().parent.parent.parent / "dataset" / "questions.json"


def test_render_per_framework_produces_markdown():
    rf = load_result_file(_FIXTURE)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    md = render_per_framework(rf.framework, rf.model, agg, scored)
    assert "# " in md
    assert "overall" in md.lower() or "Overall" in md
    assert "math-001" in md


def test_render_comparison_produces_table(tmp_path):
    # one fake framework summary
    summaries = [{"framework": "sample", "model": "m", "overall": 0.9, "by_subject": {"math": 0.9}, "by_type": {"multiple_choice": 1.0}, "count": 2}]
    md = render_comparison(summaries)
    assert "| framework |" in md
    assert "sample" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/grader/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `grader/report.py`**

```python
"""Render per-framework and cross-framework markdown reports."""
from __future__ import annotations

from datetime import datetime, timezone

from grader.models import Score

_SUBJECTS = ["chinese", "math", "english", "physics", "chemistry", "biology", "history", "geography"]


def render_per_framework(framework: str, model: str, agg: dict, scored: list[Score]) -> str:
    overall = agg["overall"]
    lines = [
        f"# {framework} — Results",
        f"",
        f"- **Model:** {model}",
        f"- **Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- **Questions scored:** {agg.get('count', 0)}",
        f"- **Overall (normalized):** {overall:.1%}",
        f"",
        "## By subject",
        "",
        "| Subject | Score |",
        "|---|---|",
    ]
    for subj in _SUBJECTS:
        val = agg["by_subject"].get(subj)
        if val is not None:
            lines.append(f"| {subj} | {val:.1%} |")
    lines += ["", "## By question type", "", "| Type | Score |", "|---|---|"]
    for t, v in agg["by_type"].items():
        lines.append(f"| {t} | {v:.1%} |")
    lines += ["", "## Per-question detail", "", "| Question | Earned/Max | Correct | Grader |", "|---|---|---|---|"]
    for s in scored:
        lines.append(f"| {s.question_id} | {int(s.earned)}/{s.max_points} | {'✓' if s.correct else '✗'} | {s.grader} |")
    return "\n".join(lines) + "\n"


def render_comparison(summaries: list[dict]) -> str:
    lines = [
        "# Framework Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Overall",
        "",
        "| framework | model | overall | questions |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| {s['framework']} | {s.get('model','?')} | {s['overall']:.1%} | {s.get('count','?')} |")

    lines += ["", "## By subject", "", "| framework | " + " | ".join(_SUBJECTS) + " |", "|---|" + "---|" * len(_SUBJECTS)]
    for s in summaries:
        row = [s["framework"]]
        for subj in _SUBJECTS:
            v = s.get("by_subject", {}).get(subj)
            row.append(f"{v:.1%}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/grader/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add grader/report.py tests/grader/test_report.py
git commit -m "feat(grader): add markdown report renderer"
```

---

## Task 14: Grader CLI (`grade.py`, `grade_all.py`)

**Files:**
- Create: `grader/grade.py`
- Create: `grader/grade_all.py`
- Create: `grader/requirements.txt`

- [ ] **Step 1: Write `grader/requirements.txt`**

```
pydantic>=2.6
jsonschema>=4.21
click>=8.1
python-dotenv>=1.0
openai>=1.30
```

- [ ] **Step 2: Write `grader/grade.py`**

```python
"""CLI: grade a single framework result file.

Usage:
    python -m grader.grade --result <result.json> --dataset <questions.json> [--judge llm]
"""
from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from grader.loaders import load_result_file
from grader.report import render_per_framework
from grader.score import aggregate, score_result_file

load_dotenv()


@click.command()
@click.option("--result", "result_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--judge", type=click.Choice(["rubric", "llm"]), default="rubric",
              help="rubric (default, deterministic) or llm (LLM-as-judge)")
@click.option("--report-dir", default=None, type=click.Path(),
              help="Optional dir to write a markdown report")
def cli(result_path: str, dataset_path: str, judge: str, report_dir: str | None) -> None:
    rf = load_result_file(result_path)
    scored = score_result_file(rf, dataset_path=dataset_path, judge=judge)
    agg = aggregate(scored)
    click.echo(f"{rf.framework}: overall={agg['overall']:.1%} ({agg['count']} questions)")
    if report_dir:
        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        md = render_per_framework(rf.framework, rf.model, agg, scored)
        out = out_dir / f"{rf.framework}.md"
        out.write_text(md)
        click.echo(f"report -> {out}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 3: Write `grader/grade_all.py`**

```python
"""CLI: grade every frameworks/*/result/results.json and write a comparison report.

Usage:
    python -m grader.grade_all [--judge llm] [--report-dir reports]
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import click
from dotenv import load_dotenv

from grader.loaders import load_result_file
from grader.report import render_comparison, render_per_framework
from grader.score import aggregate, score_result_file

load_dotenv()

_REPO = Path(__file__).resolve().parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_FRAMEWORKS = _REPO / "frameworks"


def _discover_results() -> list[Path]:
    if not _FRAMEWORKS.exists():
        return []
    return sorted(_FRAMEWORKS.glob("*/result/results.json"))


@click.command()
@click.option("--judge", type=click.Choice(["rubric", "llm"]), default="rubric")
@click.option("--report-dir", default="reports", type=click.Path())
def cli(judge: str, report_dir: str) -> None:
    results = _discover_results()
    if not results:
        click.echo("No frameworks/*/result/results.json found.")
        raise SystemExit(0)

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summaries = []
    for p in results:
        rf = load_result_file(p)
        scored = score_result_file(rf, dataset_path=_DATASET, judge=judge)
        agg = aggregate(scored)
        summaries.append({
            "framework": rf.framework,
            "model": rf.model,
            **agg,
        })
        # per-framework report
        md = render_per_framework(rf.framework, rf.model, agg, scored)
        (out_dir / f"{rf.framework}-{ts}.md").write_text(md)
        click.echo(f"{rf.framework}: {agg['overall']:.1%}")

    comparison = render_comparison(summaries)
    comp_path = out_dir / f"comparison-{ts}.md"
    comp_path.write_text(comparison)
    click.echo(f"comparison -> {comp_path}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Smoke-test the CLI against the fixture (not a real framework)**

Run from repo root:
```bash
python -m grader.grade --result tests/fixtures/sample_result.json --dataset dataset/questions.json --report-dir /tmp/rep
cat /tmp/rep/sample.md
```
Expected: prints `sample: overall=...%` and writes a markdown file with a per-question table.

- [ ] **Step 5: Commit**

```bash
git add grader/grade.py grader/grade_all.py grader/requirements.txt
git commit -m "feat(grader): add grade and grade_all CLIs"
```

---

## Task 15: Top-level root README

**Files:**
- Modify: `README.md` (currently a single placeholder line)

- [ ] **Step 1: Replace `README.md` content**

```markdown
# Awesome Agent Design Comparison

A benchmark and design-comparison project that evaluates how different LLM agent
frameworks solve the same exam task — the Chinese **Zhongkao** (中考, the senior
secondary school entrance exam).

We build a shared dataset of 50 translated exam questions spanning 8 subjects,
then implement the same solver independently with four popular agent frameworks.
A shared grader scores every implementation against the same rubric, so the
results are directly comparable.

## Why

- **Benchmark accuracy** — which framework answers most correctly?
- **Compare design patterns** — how does each framework structure the same task?
- **Showcase** — idiomatic reference implementations you can study.
- **Reproducibility** — one dataset, one grader, one report.

## The Dataset

50 Zhongkao questions, translated to English, covering:
Chinese, Math, English, Physics, Chemistry, Biology, History, Geography.

Question types: multiple choice (32) + short answer (18). See [`dataset/`](dataset/).

| Subject | MC | Short | Total |
|---|---|---|---|
| chinese | 5 | 3 | 8 |
| math | 4 | 4 | 8 |
| english | 5 | 3 | 8 |
| physics | 4 | 2 | 6 |
| chemistry | 4 | 2 | 6 |
| biology | 4 | 2 | 6 |
| history | 3 | 1 | 4 |
| geography | 3 | 1 | 4 |
| **Total** | **32** | **18** | **50** |

## Frameworks Compared

| Framework | Style | Folder |
|---|---|---|
| OpenAI Agents SDK | minimal agent + tools, async | `frameworks/openai-agents/` |
| Google ADK | hierarchical multi-agent orchestration | `frameworks/google-adk/` |
| LangGraph | explicit state-graph workflows | `frameworks/langgraph/` |
| CrewAI | role-based collaborative crews | `frameworks/crewai/` |

Each is fully self-contained — its own deps, venv, and `run.sh`.

> **Status:** the shared foundation (dataset, contract, grader) is implemented.
> Framework implementations land in subsequent phases.

## Quick Start

```bash
# 1. Install dev/test deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 2. Validate the dataset
pytest tests/test_dataset.py -v

# 3. Run the full benchmark (once frameworks exist)
./run_all.sh
```

## How It Works

```
dataset/questions.json  ──┐
                          ├──> framework run.py ──> frameworks/<name>/result/results.json
contract/*.schema.json ───┤                                     │
                          └──> grader ──────────────────────────> reports/
```

Shared dataset → each framework writes `result/results.json` in a fixed schema →
the shared grader scores it → a markdown report.

## Project Layout

```
dataset/      shared 50-question exam (read-only)
contract/     JSON Schemas + validator (read-only)
frameworks/   one self-contained folder per agent framework
grader/       shared hybrid grader (deterministic default, optional LLM-judge)
reports/      generated comparison reports (gitignored)
docs/         design spec + implementation plans
```

## Grading

Hybrid:

- **Deterministic rubric** (default) — fully reproducible, no API calls.
- **LLM-as-judge** (`--judge llm`) — more nuanced, costs tokens, non-deterministic.

Scores are normalized to `[0,1]` per question (earned / max_points), then
averaged across questions, subjects, and types.

```bash
python -m grader.grade --result <result.json> --dataset dataset/questions.json
python -m grader.grade_all --report-dir reports
```

## Extending

Add a framework: drop a new `frameworks/<name>/` folder with a `run.sh` +
`run.py` that honors [`contract/result.schema.json`](contract/result.schema.json).
`run_all.sh` picks it up automatically via `frameworks/*/`.

## License

MIT (code). Dataset questions are based on/inspired by real past Zhongkao
papers, translated to English; see `source` fields for attribution.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: write English project README"
```

---

## Task 16: Phase 1 verification

- [ ] **Step 1: Run the full test suite**

Run:
```bash
python -m pytest -v
```
Expected: all tests pass (≈ 35+ tests across contract, dataset, grader).

- [ ] **Step 2: Validate the dataset end-to-end**

Run:
```bash
# each question validates
python -m pytest tests/test_dataset.py -v
# grader works on the sample fixture
python -m grader.grade --result tests/fixtures/sample_result.json --dataset dataset/questions.json
```
Expected: dataset tests pass; grader prints an overall percentage without error.

- [ ] **Step 3: Confirm directory tree**

Run:
```bash
find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path '*/__pycache__/*' | sort
```
Expected: includes `contract/`, `dataset/questions.json`, `grader/`, `tests/`,
`requirements-dev.txt`, `README.md`, and the spec/plan under `docs/`.

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git status
# if anything staged:
git commit -m "chore: phase 1 foundation complete"
```

---

## Self-Review

**1. Spec coverage:** Every foundation requirement in the design spec maps to a task:
- Repo layout → Task 1 (gitignore + deps)
- Contract schemas → Tasks 2, 3
- Dataset → Task 6
- Result schema → Task 3
- Grader (exact + rubric + llm + orchestrator + report + CLI) → Tasks 7–14
- Per-framework `run.sh` / `run_all.sh` → **deferred to Phase 2+** (frameworks don't exist yet; `run_all.sh` is only meaningful once at least one framework lands). This is correct: Phase 1 is the shared foundation only.
- README → Task 15

**2. Placeholder scan:** No "TBD"/"TODO" in steps. The dataset (Task 6) contains the full 50 questions verbatim, not a stub. All test code is complete.

**3. Type consistency:** `Score` fields (`question_id`, `earned`, `max_points`, `normalized`, `correct`, `per_criterion`, `grader`) are consistent across `models.py`, all graders, `score.py`, and `report.py`. `Grader.score(question, answer) -> Score` signature is identical in `base.py`, `exact.py`, `rubric.py`, `llm_judge.py`. `aggregate()` keys (`overall`, `by_subject`, `by_type`, `count`) match between `score.py` and `report.py`. ✓

**4. Scope check:** This phase produces working, testable software on its own — the dataset, contract, and grader all function and are verified via `pytest` and the CLI smoke test. Frameworks land in Phase 2+. ✓
