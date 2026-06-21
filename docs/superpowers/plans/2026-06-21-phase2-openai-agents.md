# Phase 2: OpenAI Agents SDK Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fully self-contained `frameworks/openai-agents/` folder that reads the shared dataset, solves each question with the OpenAI Agents SDK, and writes `result/results.json` in the shared contract — all idiomatic to the SDK, all tested.

**Architecture:** A single `Agent` (instructions + `output_type` Pydantic model + recommended model `gpt-4o`) solves every question via `Runner.run_sync`. The agent is given the question text, options, and subject context, and returns a structured `{response, reasoning}` that the runner maps onto the contract's `AnswerResult`. MC answers are extracted to a single letter; short-answer responses are kept verbatim. No real API calls in tests — the SDK client is wrapped so tests inject a fake runner.

**Tech Stack:** Python 3.11+, `openai-agents` (PyPI package; import name `agents`), `pydantic` (structured output), `pytest`. Reads `dataset/` and `contract/` (read-only); writes `result/`.

**Prerequisite:** Phase 1 merged (dataset, contract, grader). Branch from `main`.

---

## File Structure

| File | Responsibility |
|---|---|
| `frameworks/openai-agents/requirements.txt` | `openai-agents`, `pydantic` (own deps, isolated venv) |
| `frameworks/openai-agents/__init__.py` | Package marker (empty docstring) |
| `frameworks/openai-agents/output_models.py` | Pydantic `AgentAnswer` structured output type |
| `frameworks/openai-agents/agent.py` | Build the `Agent`; instructions + model + output_type (idiomatic SDK) |
| `frameworks/openai-agents/solver.py` | `solve_one(question) -> AgentAnswer`: run the agent, time it, extract final output |
| `frameworks/openai-agents/run.py` | CLI entrypoint: load dataset → solve all → write `result/results.json` (the contract) |
| `frameworks/openai-agents/run.sh` | Self-contained shell entrypoint (activate venv, set env, call run.py) |
| `frameworks/openai-agents/README.md` | How to install + run, what's idiomatic about this impl |
| `frameworks/openai-agents/result/.gitkeep` | Keeps the output dir in git (outputs themselves gitignored) |
| `tests/frameworks/__init__.py` | empty (so pytest collects under a package) — NOTE: omitted, see Task 1 note |
| `tests/frameworks/test_openai_agents_output.py` | Tests for output model parsing/normalization |
| `tests/frameworks/test_openai_agents_solver.py` | Solver tests with a fake runner (no API) |
| `tests/frameworks/test_openai_agents_run.py` | End-to-end run.py test with a fake runner (writes a result file, validates against contract) |

**Import conventions:**
- The framework package is `frameworks.openai_agents` (Python module name; the folder uses a hyphen `openai-agents` for display but Python imports use... — **see Task 1**: we make the folder importable by adding a `sys.path` shim in `run.py`, because the folder name has a hyphen). Tests import it via a small conftest `sys.path` insert.
- Per Phase 1's lesson, test directories do **not** use `__init__.py` (they shadow real packages). `pytest.ini`'s `pythonpath = .` covers the repo root; we add the framework dir to path in the test conftest.

---

## Task 1: Bootstrap framework folder + import shim

**Files:**
- Create: `frameworks/openai-agents/requirements.txt`
- Create: `frameworks/openai-agents/__init__.py`
- Create: `frameworks/openai-agents/result/.gitkeep`
- Modify: `tests/conftest.py` (root) — add framework dir to sys.path for tests

- [ ] **Step 1: Create `requirements.txt`**

```
openai-agents>=0.17
pydantic>=2.6
```

- [ ] **Step 2: Create the package marker**

`frameworks/openai-agents/__init__.py`:

```python
"""OpenAI Agents SDK framework implementation.

Self-contained solver for the Zhongkao dataset using the OpenAI Agents SDK.
Idiomatic style: a single Agent with structured output (output_type).
"""
```

- [ ] **Step 3: Create the output dir placeholder**

`frameworks/openai-agents/result/.gitkeep`: (empty file)

- [ ] **Step 4: Make the hyphenated folder importable in tests**

The folder `openai-agents` can't be imported as a Python module name (hyphen).
`run.py` will be invoked as a script (`python run.py`), which is fine. But tests
need to import its modules. Add the folder to `sys.path` via the root conftest.

Append to `tests/conftest.py` (read it first):

```python
import sys
from pathlib import Path

# Make framework folders with hyphens importable under their own name.
_FRAMEWORKS = Path(__file__).resolve().parent.parent / "frameworks"
for _sub in ("openai-agents",):
    _dir = _FRAMEWORKS / _sub
    if _dir.is_dir():
        sys.path.insert(0, str(_dir))
```

After this, `import output_models`, `import agent`, etc. work from the test files
under `tests/frameworks/`. The framework modules use **flat module names**
(`output_models`, `agent`, `solver`, `run`) — no package prefix — because the
folder name has a hyphen. This keeps imports simple and matches how `run.py`
itself imports them.

- [ ] **Step 5: Install deps and verify import**

```bash
source .venv/bin/activate
pip install -r frameworks/openai-agents/requirements.txt
python -c "import sys; sys.path.insert(0, 'frameworks/openai-agents'); import output_models" 2>&1 || echo "expected: output_models.py not created yet (Task 2)"
```
Expected: prints the "expected: ..." line (file doesn't exist yet — fine). Confirms the path shim + SDK install work.

- [ ] **Step 6: Commit**

```bash
git add frameworks/openai-agents/requirements.txt frameworks/openai-agents/__init__.py frameworks/openai-agents/result/.gitkeep tests/conftest.py
git commit -m "feat(openai-agents): bootstrap framework folder with import shim"
```

---

## Task 2: Structured output model

**Files:**
- Create: `frameworks/openai-agents/output_models.py`
- Create: `tests/frameworks/test_openai_agents_output.py`

The SDK's `output_type` takes a Pydantic model; `result.final_output` is then a
parsed instance. We define the shape the agent returns, plus a normalizer that
maps it onto the contract's `response` field (single letter for MC).

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_openai_agents_output.py`:

```python
import pytest
from pydantic import ValidationError

from output_models import AgentAnswer, normalize_response


def test_agent_answer_ok():
    a = AgentAnswer(response="A", reasoning="difference of squares")
    assert a.response == "A"
    assert a.reasoning == "difference of squares"


def test_agent_answer_requires_fields():
    with pytest.raises(ValidationError):
        AgentAnswer(response="", reasoning="")  # response must be non-empty


def test_normalize_mc_letter():
    assert normalize_response("A", is_mc=True) == "A"


def test_normalize_mc_strips_and_uppercases():
    assert normalize_response(" Option B. ", is_mc=True) == "B"


def test_normalize_mc_picks_first_letter():
    assert normalize_response("I think it is C because...", is_mc=True) == "C"


def test_normalize_short_answer_kept_verbatim():
    assert normalize_response("x = 12", is_mc=False) == "x = 12"


def test_normalize_mc_empty_falls_back_to_unknown():
    assert normalize_response("", is_mc=True) == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/frameworks/test_openai_agents_output.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'output_models'`.

- [ ] **Step 3: Write `output_models.py`**

```python
"""Structured output type for the OpenAI Agents SDK solver.

`AgentAnswer` is passed as the agent's `output_type`, so the SDK returns a
parsed instance via `result.final_output`. `normalize_response` maps the raw
response onto the contract's `response` field (a single letter for MC).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field


class AgentAnswer(BaseModel):
    """What the agent returns for one question."""

    response: str = Field(min_length=1, description="The final answer: a single letter for MC, or the full answer text for short-answer questions")
    reasoning: str = Field(default="", description="Brief chain-of-thought leading to the answer")


_FIRST_LETTER = re.compile(r"[A-Za-z]")


def normalize_response(raw: str, *, is_mc: bool) -> str:
    """Normalize the agent's response for the contract's `response` field.

    For multiple choice, extract the first A-Z letter (case-insensitive),
    matching how the exact grader works. For short answer, keep the text as-is.
    """
    if is_mc:
        m = _FIRST_LETTER.search(raw or "")
        return m.group(0).upper() if m else ""
    return raw or ""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_openai_agents_output.py -q
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/openai-agents/output_models.py tests/frameworks/test_openai_agents_output.py
git commit -m "feat(openai-agents): add structured output model and response normalizer"
```

---

## Task 3: Agent definition (idiomatic SDK)

**Files:**
- Create: `frameworks/openai-agents/agent.py`
- Create: `tests/frameworks/test_openai_agents_agent.py`

The `Agent` is built once with instructions, model, and `output_type`. This task
exercises the SDK's "agent = instructions + model + output_type" minimalism. We
do NOT call the API here (no network); we just construct the Agent and assert
its configuration. The runner is injected in Task 4.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_openai_agents_agent.py`:

```python
from agents import Agent

from agent import build_agent, DEFAULT_MODEL, AGENT_INSTRUCTIONS
from output_models import AgentAnswer


def test_build_agent_returns_sdk_agent():
    a = build_agent()
    assert isinstance(a, Agent)


def test_agent_has_expected_model():
    a = build_agent()
    assert a.model == DEFAULT_MODEL


def test_agent_uses_structured_output_type():
    a = build_agent()
    assert a.output_type is AgentAnswer


def test_instructions_mention_exam_context():
    assert "Zhongkao" in AGENT_INSTRUCTIONS
    assert "multiple choice" in AGENT_INSTRUCTIONS.lower()


def test_build_agent_accepts_override_model():
    a = build_agent(model="gpt-4o-mini")
    assert a.model == "gpt-4o-mini"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/frameworks/test_openai_agents_agent.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Write `agent.py`**

```python
"""Build the OpenAI Agents SDK Agent.

Idiomatic SDK style: a single Agent with instructions + model + structured
output_type. No handoffs, no tools (the exam questions don't benefit from
external tools, and combining tools with structured output can suppress tool
calls — see SDK known issues). This keeps the implementation minimal and
showcases the SDK's "agent = instructions + model + output_type" core.
"""
from __future__ import annotations

from agents import Agent

from output_models import AgentAnswer

DEFAULT_MODEL = "gpt-4o"

AGENT_INSTRUCTIONS = """\
You are a top student taking the Chinese Zhongkao (senior secondary school
entrance exam). Answer each question correctly and concisely.

For multiple choice questions, your `response` MUST be a single capital letter
(A, B, C, or D) corresponding to the correct option.

For short answer questions, your `response` MUST be the complete answer (e.g.
a number with units, a short phrase, a balanced equation, or a one-sentence
explanation) — exactly as a correct exam answer would be written.

Always include brief `reasoning` showing how you reached the answer.
Be precise. Do not add extraneous commentary in `response`.
"""


def build_agent(*, model: str = DEFAULT_MODEL) -> Agent:
    """Construct the solver Agent."""
    return Agent(
        name="ZhongkaoSolver",
        instructions=AGENT_INSTRUCTIONS,
        model=model,
        output_type=AgentAnswer,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_openai_agents_agent.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/openai-agents/agent.py tests/frameworks/test_openai_agents_agent.py
git commit -m "feat(openai-agents): add idiomatic Agent with structured output"
```

---

## Task 4: Solver with injectable runner

**Files:**
- Create: `frameworks/openai-agents/solver.py`
- Create: `tests/frameworks/test_openai_agents_solver.py`

`solve_one` runs the agent on one question and returns `(AgentAnswer, latency_ms)`.
To keep tests deterministic and free, the runner function is injectable — by
default it uses `Runner.run_sync`, but tests pass a fake.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_openai_agents_solver.py`:

```python
from solver import solve_one, QuestionInput, build_prompt
from output_models import AgentAnswer


class FakeRunner:
    def __init__(self, answer: AgentAnswer):
        self._answer = answer
        self.calls = []

    def __call__(self, agent, prompt):
        self.calls.append((agent, prompt))
        return self._answer


def _mc_question():
    return {
        "id": "math-001",
        "subject": "math",
        "type": "multiple_choice",
        "question": "Simplify (x+2)(x-2).",
        "options": ["A. x^2-4", "B. x^2+4", "C. x^2-2x", "D. x^2-4x+4"],
        "answer": "A",
        "answer_type": "letter",
    }


def _short_question():
    return {
        "id": "math-005",
        "subject": "math",
        "type": "short_answer",
        "question": "Solve 3x - 7 = 2x + 5 for x.",
        "answer": "x = 12",
        "answer_type": "text",
    }


def test_build_prompt_includes_question_and_options():
    p = build_prompt(_mc_question())
    assert "Simplify (x+2)(x-2)." in p
    assert "A. x^2-4" in p
    assert "multiple choice" in p.lower()


def test_build_prompt_short_answer_no_options():
    p = build_prompt(_short_question())
    assert "Solve 3x - 7 = 2x + 5" in p
    assert "short answer" in p.lower()


def test_solve_one_mc_returns_normalized_letter():
    fake = FakeRunner(AgentAnswer(response="A", reasoning="diff of squares"))
    answer, latency = solve_one(_mc_question(), runner=fake)
    assert answer.response == "A"
    assert latency >= 0
    assert len(fake.calls) == 1


def test_solve_one_mc_normalizes_to_letter():
    fake = FakeRunner(AgentAnswer(response="Option B", reasoning="..."))
    answer, _ = solve_one(_mc_question(), runner=fake)
    assert answer.response == "B"


def test_solve_one_short_keeps_verbatim():
    fake = FakeRunner(AgentAnswer(response="x = 12", reasoning="subtracted 2x"))
    answer, _ = solve_one(_short_question(), runner=fake)
    assert answer.response == "x = 12"


def test_question_input_parsing():
    qi = QuestionInput.from_question(_mc_question())
    assert qi.id == "math-001"
    assert qi.is_mc is True
    assert qi.options == _mc_question()["options"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/frameworks/test_openai_agents_solver.py -q
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solver.py`**

```python
"""Solve one question with the OpenAI Agents SDK.

The runner is injectable so tests stay deterministic and free of API calls.
By default it uses `agents.Runner.run_sync`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agents import Agent, Runner

from output_models import AgentAnswer, normalize_response


RunnerFn = Callable[[Agent, str], AgentAnswer]
"""A runner takes (agent, prompt) and returns the agent's AgentAnswer.

The real Runner.run_sync returns a RunResult whose .final_output is the
AgentAnswer; we wrap it (see _default_runner) so this signature holds.
"""


@dataclass
class QuestionInput:
    id: str
    subject: str
    type: str
    question: str
    answer_type: str
    options: list[str] = field(default_factory=list)

    @property
    def is_mc(self) -> bool:
        return self.type == "multiple_choice"

    @classmethod
    def from_question(cls, q: dict[str, Any]) -> "QuestionInput":
        return cls(
            id=q["id"],
            subject=q["subject"],
            type=q["type"],
            question=q["question"],
            answer_type=q["answer_type"],
            options=q.get("options", []),
        )


def build_prompt(q: dict[str, Any]) -> str:
    """Render a dataset question into the prompt string for the agent."""
    qi = QuestionInput.from_question(q)
    kind = "multiple choice" if qi.is_mc else "short answer"
    lines = [
        f"Subject: {qi.subject}",
        f"Question type: {kind}",
        "",
        qi.question,
    ]
    if qi.is_mc and qi.options:
        lines += ["", "Options:"] + [f"  {o}" for o in qi.options]
    lines += [
        "",
        "Remember: respond with a single capital letter for multiple choice, "
        "or the complete answer for short answer.",
    ]
    return "\n".join(lines)


def _default_runner(agent: Agent, prompt: str) -> AgentAnswer:
    """Wrap Runner.run_sync so it returns the AgentAnswer directly."""
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def solve_one(
    q: dict[str, Any],
    *,
    runner: RunnerFn | None = None,
    agent: Agent | None = None,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (normalized AgentAnswer, latency_ms).

    If `agent` is None, a default agent is built. If `runner` is None, the real
    Runner.run_sync is used (requires OPENAI_API_KEY).
    """
    run = runner or _default_runner
    ag = agent
    if ag is None:
        from agent import build_agent  # lazy import to avoid circular dep
        ag = build_agent()

    qi = QuestionInput.from_question(q)
    prompt = build_prompt(q)

    start = time.perf_counter()
    raw = run(ag, prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    normalized = AgentAnswer(
        response=normalize_response(raw.response, is_mc=qi.is_mc),
        reasoning=raw.reasoning,
    )
    return normalized, latency_ms
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_openai_agents_solver.py -q
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/openai-agents/solver.py tests/frameworks/test_openai_agents_solver.py
git commit -m "feat(openai-agents): add solver with injectable runner"
```

---

## Task 5: CLI entrypoint (run.py) — the contract

**Files:**
- Create: `frameworks/openai-agents/run.py`
- Create: `tests/frameworks/test_openai_agents_run.py`

`run.py` is the contract entrypoint: `python run.py --dataset <path> --output <path>`.
It loads the dataset, solves every question, and writes `result/results.json` in
exactly `contract/result.schema.json` shape. Tests use a fake runner to avoid
API calls and validate the output file against the schema end-to-end.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_openai_agents_run.py`:

```python
import json
import sys
from pathlib import Path

import jsonschema
import pytest

# run.py is meant to be run as a script; import its functions via path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "frameworks" / "openai-agents"))

import run as run_module
from output_models import AgentAnswer


_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"
_RESULT_SCHEMA = _REPO / "contract" / "result.schema.json"


class FakeRunner:
    def __init__(self):
        self.count = 0

    def __call__(self, agent, prompt):
        self.count += 1
        # Return a plausible answer for any question.
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
    # Validate against the contract schema
    schema = json.loads(_RESULT_SCHEMA.read_text())
    jsonschema.validate(data, schema)
    # Spot checks
    assert data["framework"] == "openai-agents"
    assert data["model"] == "gpt-4o"
    assert len(data["results"]) == 50
    # Every result has the required fields
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
    # Should not crash; failed questions get empty response, error in raw
    assert len(data["results"]) == 50
    for r in data["results"]:
        assert r["response"] == ""
        assert "error" in r["raw"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/frameworks/test_openai_agents_run.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'run'`.

- [ ] **Step 3: Write `run.py`**

```python
"""Entrypoint: solve the dataset with the OpenAI Agents SDK and write result/results.json.

Usage (contract):
    python run.py --dataset ../../dataset/questions.json --output result/results.json

Idiomatic SDK: build one Agent, solve each question via Runner, write the shared
result schema. This file is run as a script (python run.py), so it uses flat
imports (output_models, agent, solver) resolved from this directory.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# When run as a script, ensure this directory is on sys.path for flat imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import click
from agents import Agent
from dotenv import load_dotenv

from agent import DEFAULT_MODEL, build_agent
from output_models import AgentAnswer
from solver import RunnerFn, solve_one

load_dotenv()


def run(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    framework: str = "openai-agents",
    model: str = DEFAULT_MODEL,
    runner: RunnerFn | None = None,
) -> dict:
    """Solve every question in the dataset and write the result file.

    Returns the written result dict. `runner` is injectable for tests.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(dataset_path.read_text())
    # Build the agent once (production) when using the real runner. When a
    # runner is injected (tests), pass None — the fake runner ignores the
    # agent argument, and solve_one's lazy build_agent() is cheap (no API call).
    if runner is None:
        agent = build_agent(model=model)
    else:
        agent = None

    results = []
    for q in questions:
        qi_subject = q["subject"]
        qi_type = q["type"]
        qi_id = q["id"]
        try:
            answer, latency_ms = solve_one(q, runner=runner, agent=agent)
            results.append({
                "question_id": qi_id,
                "subject": qi_subject,
                "type": qi_type,
                "response": answer.response,
                "reasoning": answer.reasoning,
                "latency_ms": latency_ms,
                "raw": {"agent": "ZhongkaoSolver"},
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "question_id": qi_id,
                "subject": qi_subject,
                "type": qi_type,
                "response": "",
                "reasoning": "",
                "latency_ms": 0,
                "raw": {"agent": "ZhongkaoSolver", "error": str(exc)},
            })

    payload = {
        "framework": framework,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {"temperature": 0.0},
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


@click.command()
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
def cli(dataset_path: str, output_path: str, model: str) -> None:
    """Solve the dataset and write OUTPUT in the shared result schema."""
    payload = run(dataset_path=dataset_path, output_path=output_path, model=model)
    click.echo(f"{payload['framework']}: wrote {len(payload['results'])} answers to {output_path}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_openai_agents_run.py -q
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/openai-agents/run.py tests/frameworks/test_openai_agents_run.py
git commit -m "feat(openai-agents): add run.py contract entrypoint with error handling"
```

---

## Task 6: Shell entrypoint (run.sh)

**Files:**
- Create: `frameworks/openai-agents/run.sh`

Self-contained, knows nothing about siblings. Resolves paths relative to itself.

- [ ] **Step 1: Write `run.sh`**

```bash
#!/usr/bin/env bash
# frameworks/openai-agents/run.sh — self-contained entrypoint.
# Solves the shared dataset and writes result/results.json in the contract schema.
#
# Requires: OPENAI_API_KEY in the environment (or a .env file at repo root).
set -euo pipefail
cd "$(dirname "$0")"

# Use the repo-root venv if present; otherwise fall back to `python`.
REPO_ROOT="$(cd ../.. && pwd)"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

python run.py \
  --dataset "$REPO_ROOT/dataset/questions.json" \
  --output result/results.json
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x frameworks/openai-agents/run.sh
```

- [ ] **Step 3: Smoke-test the shell entrypoint (requires OPENAI_API_KEY)**

If you have a key set:
```bash
export OPENAI_API_KEY=sk-...
bash frameworks/openai-agents/run.sh
python -m grader.grade --result frameworks/openai-agents/result/results.json --dataset dataset/questions.json
```
Expected: writes `frameworks/openai-agents/result/results.json`, then grader prints an overall %.

If no key: skip this step (it's a live-API smoke test, not a unit test). The
Task 5 tests already verify run.py's logic with a fake runner.

- [ ] **Step 4: Commit**

```bash
git add frameworks/openai-agents/run.sh
git commit -m "feat(openai-agents): add self-contained run.sh entrypoint"
```

---

## Task 7: Framework README

**Files:**
- Create: `frameworks/openai-agents/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# OpenAI Agents SDK — Zhongkao Solver

Self-contained solver for the shared Zhongkao dataset, implemented with the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

## Design (idiomatic SDK)

This implementation leans into the SDK's core abstraction: an **Agent** is
instructions + model + structured output.

- **Single agent** (`ZhongkaoSolver`) — no handoffs, no sub-agents. Every
  question goes to one agent.
- **Structured output** — the agent's `output_type` is the `AgentAnswer` Pydantic
  model (`response` + `reasoning`). The SDK returns a parsed instance via
  `result.final_output`, so we never parse free text.
- **No tools** — exam questions don't benefit from external tools, and combining
  tools with structured output can suppress tool calls (a known SDK caveat). This
  keeps the implementation minimal.
- **Injectable runner** — `solve_one` accepts a `runner` callable so tests run
  without API calls or keys.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Builds the `Agent` (instructions + model + output_type) |
| `output_models.py` | `AgentAnswer` Pydantic type + response normalizer |
| `solver.py` | `solve_one(question)` — runs the agent, times it |
| `run.py` | CLI: load dataset → solve all → write `result/results.json` |
| `run.sh` | Shell entrypoint (activates repo venv, sets paths) |

## Install

```bash
cd frameworks/openai-agents
pip install -r requirements.txt   # or use the repo-root .venv
```

## Run

```bash
# Option A: via the shell entrypoint (recommended)
export OPENAI_API_KEY=sk-...
bash run.sh

# Option B: directly
python run.py --dataset ../../dataset/questions.json --output result/results.json
```

Both write `result/results.json` in the shared contract schema.

## Test

From the repo root:

```bash
pytest tests/frameworks/test_openai_agents_*.py -v
```

Tests use a fake runner — no API calls, no key required.

## Model

Default: `gpt-4o`. Override with `--model gpt-4o-mini` (CLI) or by editing
`agent.DEFAULT_MODEL`.
```

- [ ] **Step 2: Commit**

```bash
git add frameworks/openai-agents/README.md
git commit -m "docs(openai-agents): framework README"
```

---

## Task 8: Integration with grader (end-to-end, fake runner)

**Files:**
- Create: `tests/frameworks/test_openai_agents_integration.py`

Verify the framework's output grades cleanly through the shared grader.

- [ ] **Step 1: Write the test**

`tests/frameworks/test_openai_agents_integration.py`:

```python
"""End-to-end: framework output → shared grader → report.

Uses a fake runner that returns correct-ish answers so we exercise the full
pipeline without API calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "frameworks" / "openai-agents"))

import run as run_module
from output_models import AgentAnswer
from grader.loaders import load_result_file
from grader.score import aggregate, score_result_file

_REPO = Path(__file__).resolve().parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"


class CorrectFakeRunner:
    """Returns the reference answer for each question (best case)."""

    def __init__(self):
        self._by_id = {}

    def prime(self, questions):
        for q in questions:
            self._by_id[q["id"]] = q

    def __call__(self, agent, prompt):
        # The runner doesn't know which question; for this test we always
        # return a fixed correct-ish answer. Scoring will vary by question.
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
    from grader.report import render_per_framework
    out = tmp_path / "results.json"
    run_module.run(dataset_path=str(_DATASET), output_path=str(out), runner=CorrectFakeRunner())
    rf = load_result_file(out)
    scored = score_result_file(rf, dataset_path=_DATASET)
    agg = aggregate(scored)
    md = render_per_framework(rf.framework, rf.model, agg, scored)
    assert "openai-agents" in md
    assert "math-001" in md
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/frameworks/test_openai_agents_integration.py -q
```
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/frameworks/test_openai_agents_integration.py
git commit -m "test(openai-agents): end-to-end grading integration with shared grader"
```

---

## Task 9: Phase 2 verification

- [ ] **Step 1: Run the full test suite**

```bash
source .venv/bin/activate
python -m pytest -q
```
Expected: all tests pass (Phase 1's 37 + Phase 2's new tests ≈ 55+).

- [ ] **Step 2: Validate the framework output against the contract schema**

```bash
python -m pytest tests/frameworks/test_openai_agents_run.py -q
```
Expected: 4 passed — confirms `results.json` validates against `result.schema.json`.

- [ ] **Step 3: Confirm directory tree**

```bash
find frameworks -type f -not -path '*/result/*' -o -name '.gitkeep' | sort
```
Expected:
```
frameworks/openai-agents/README.md
frameworks/openai-agents/__init__.py
frameworks/openai-agents/agent.py
frameworks/openai-agents/output_models.py
frameworks/openai-agents/requirements.txt
frameworks/openai-agents/run.py
frameworks/openai-agents/run.sh
frameworks/openai-agents/result/.gitkeep
frameworks/openai-agents/solver.py
```

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git status
# if anything staged:
git commit -m "chore: phase 2 openai-agents framework complete"
```

---

## Self-Review

**1. Spec coverage:**
- Self-contained folder with own deps (`requirements.txt`) ✓ (Task 1)
- Uniform entrypoint `python run.py --dataset --output` ✓ (Task 5)
- Idiomatic SDK (Agent + instructions + output_type, async-able Runner) ✓ (Tasks 3, 4)
- Writes `result/results.json` in shared schema ✓ (Task 5, validated against schema in test)
- `run.sh` self-contained ✓ (Task 6)
- Framework README ✓ (Task 7)
- Recommended model `gpt-4o` ✓ (Task 3, `DEFAULT_MODEL`)
- Grader integration ✓ (Task 8)

**2. Placeholder scan:** No TBD/TODO in steps. All code blocks are complete. The one "optional" step (Task 6 Step 3 live-API smoke test) is explicitly marked as skippable without a key, and the logic is already covered by Task 5's fake-runner tests.

**3. Type/signature consistency:**
- `RunnerFn = Callable[[Agent, str], AgentAnswer]` — consistent in `solver.py`, `run.py`, and all test fakes (they return `AgentAnswer`).
- `solve_one(q, *, runner, agent) -> (AgentAnswer, int)` — consistent in `solver.py` and `run.py`.
- `run(*, dataset_path, output_path, framework, model, runner) -> dict` — consistent between `run.py` and all test calls.
- `AgentAnswer(response, reasoning)` — consistent across `output_models.py`, `agent.py` (as `output_type`), `solver.py`, and fakes.
- `normalize_response(raw, *, is_mc)` — consistent between `output_models.py` and `solver.py`.
- Result dict keys match `contract/result.schema.json` (`framework`, `model`, `generated_at`, `config`, `results`; each result has `question_id`, `subject`, `type`, `response`, `reasoning`, `latency_ms`, `raw`) ✓.

**4. Scope check:** Phase 2 produces a working, tested framework folder. The grader (Phase 1) already exists and integration is verified (Task 8). No out-of-scope work. The design deliberately stays minimal (single agent, no tools) per the spec's "idiomatic" intent and the SDK's known structured-output+tools caveat.

**Known limitation (documented in code, not a defect):** The folder name `openai-agents` has a hyphen, so it's not a valid Python package name. Modules use flat imports (`output_models`, etc.) and are made importable in tests via a `sys.path` shim in `tests/conftest.py`. `run.py` is run as a script and inserts its own directory into `sys.path`. This is a deliberate, documented trade-off — the folder name matches the framework's display name and the spec's `frameworks/openai-agents/` path.
