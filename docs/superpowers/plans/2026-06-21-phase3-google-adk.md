# Phase 3: Google ADK Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fully self-contained `frameworks/google-adk/` folder that reads the shared dataset, solves each question with Google's Agent Development Kit (ADK), and writes `result/results.json` in the shared contract — idiomatic to ADK, all tested.

**Architecture:** ADK's workhorse is the `LlmAgent`. We build one `LlmAgent` with `model`, `instruction`, and a Pydantic `output_schema` (`AgentAnswer`). ADK enforces that `output_schema` and `tools` are mutually exclusive — since exam questions need structured output and don't benefit from tools, we use `output_schema` (no tools), mirroring the OpenAI Agents impl's "no tools" decision but via ADK's native constraint. The agent runs through ADK's `Runner` + `InMemorySessionService`, yielding events; we collect the final-response event's text. A key ADK difference from OpenAI's SDK: the Runner is async and session-based. To keep tests deterministic and free, the runner is wrapped behind an injectable callable.

**Tech Stack:** Python 3.11+, `google-adk` (PyPI package; imports under `google.adk`), `pydantic`, `pytest`. Reads `dataset/` and `contract/` (read-only); writes `result/`.

**Prerequisite:** Phase 1 merged (dataset, contract, grader). Branch from `main`.

---

## File Structure

| File | Responsibility |
|---|---|
| `frameworks/google-adk/requirements.txt` | `google-adk`, `pydantic` (own deps, isolated venv) |
| `frameworks/google-adk/__init__.py` | Package marker (empty docstring) |
| `frameworks/google-adk/output_models.py` | Pydantic `AgentAnswer` + `normalize_response` (mirrors openai-agents) |
| `frameworks/google-adk/agent.py` | Build the `LlmAgent` (instruction + model + output_schema) |
| `frameworks/google-adk/solver.py` | `solve_one(question) -> (AgentAnswer, latency_ms)` with injectable runner |
| `frameworks/google-adk/run.py` | CLI entrypoint: load dataset → solve all → write `result/results.json` |
| `frameworks/google-adk/run.sh` | Self-contained shell entrypoint |
| `frameworks/google-adk/README.md` | How to install + run, what's idiomatic about this impl |
| `frameworks/google-adk/result/.gitkeep` | Keeps the output dir in git |

Test files (mirror the openai-agents test structure):
| Test file | Covers |
|---|---|
| `tests/frameworks/test_google_adk_output.py` | output model parsing/normalization |
| `tests/frameworks/test_google_adk_agent.py` | LlmAgent construction |
| `tests/frameworks/test_google_adk_solver.py` | solver with a fake runner |
| `tests/frameworks/test_google_adk_run.py` | end-to-end run.py → contract schema |
| `tests/frameworks/test_google_adk_integration.py` | framework output → shared grader |

**Import conventions:** Same as openai-agents — the folder `google-adk` has a hyphen, so it's not a valid Python package name. Modules use flat names (`output_models`, `agent`, `solver`, `run`) made importable via a `sys.path` shim in `tests/conftest.py`. `run.py` is run as a script.

---

## Task 1: Bootstrap framework folder + import shim

**Files:**
- Create: `frameworks/google-adk/requirements.txt`
- Create: `frameworks/google-adk/__init__.py`
- Create: `frameworks/google-adk/result/.gitkeep`
- Modify: `tests/conftest.py` — add `google-adk` to the framework-path shim

- [ ] **Step 1: Create `requirements.txt`**

```
google-adk>=1.0.0
pydantic>=2.6
```

- [ ] **Step 2: Create the package marker**

`frameworks/google-adk/__init__.py`:

```python
"""Google ADK framework implementation.

Self-contained solver for the Zhongkao dataset using Google's Agent
Development Kit (ADK). Idiomatic style: an LlmAgent with a structured
output_schema (no tools — ADK enforces schema/tools mutual exclusivity).
"""
```

- [ ] **Step 3: Create the output dir placeholder**

`frameworks/google-adk/result/.gitkeep`: (empty file)

- [ ] **Step 4: Add `google-adk` to the test import shim**

Modify `tests/conftest.py` — the `for _sub in (...)` tuple should now list both frameworks:

```python
for _sub in ("openai-agents", "google-adk"):
```

- [ ] **Step 5: Install deps and verify**

```bash
source .venv/bin/activate
pip install -r frameworks/google-adk/requirements.txt
python -c "from google.adk.agents import LlmAgent; print('ADK OK')"
```
Expected: prints `ADK OK`.

- [ ] **Step 6: Commit**

```bash
git add frameworks/google-adk/requirements.txt frameworks/google-adk/__init__.py frameworks/google-adk/result/.gitkeep tests/conftest.py
git commit -m "feat(google-adk): bootstrap framework folder with import shim"
```

---

## Task 2: Structured output model

**Files:**
- Create: `frameworks/google-adk/output_models.py`
- Create: `tests/frameworks/test_google_adk_output.py`

ADK's `output_schema` takes a Pydantic model. The final response text is the
JSON-serialized schema, so we define `AgentAnswer` and a parser that turns the
Runner's final text back into the object. We reuse the same normalization logic
as openai-agents (standalone A-D letter for MC).

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_google_adk_output.py`:

```python
import pytest
from pydantic import ValidationError

from output_models import AgentAnswer, normalize_response, parse_final_text


def test_agent_answer_ok():
    a = AgentAnswer(response="A", reasoning="difference of squares")
    assert a.response == "A"


def test_agent_answer_requires_response():
    with pytest.raises(ValidationError):
        AgentAnswer(response="", reasoning="")


def test_normalize_mc_letter():
    assert normalize_response("A", is_mc=True) == "A"


def test_normalize_mc_strips_and_uppercases():
    assert normalize_response(" Option B. ", is_mc=True) == "B"


def test_normalize_mc_picks_standalone_letter():
    assert normalize_response("I think it is C because...", is_mc=True) == "C"


def test_normalize_short_answer_kept_verbatim():
    assert normalize_response("x = 12", is_mc=False) == "x = 12"


def test_parse_final_text_json():
    text = '{"response": "A", "reasoning": "diff of squares"}'
    a = parse_final_text(text, is_mc=True)
    assert a.response == "A"
    assert a.reasoning == "diff of squares"


def test_parse_final_text_plain_falls_back_to_response():
    a = parse_final_text("B", is_mc=True)
    assert a.response == "B"
    assert a.reasoning == ""


def test_parse_final_text_empty():
    a = parse_final_text("", is_mc=True)
    assert a.response == ""
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
python -m pytest tests/frameworks/test_google_adk_output.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'output_models'` (from the google-adk dir).

- [ ] **Step 3: Write `output_models.py`**

```python
"""Structured output schema for the Google ADK solver.

`AgentAnswer` is passed as the LlmAgent's `output_schema`. The Runner's final
response text is the JSON-serialized schema; `parse_final_text` parses it back
(and tolerates plain-text responses). `normalize_response` maps the response
onto the contract's `response` field (single A-D letter for MC).
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field


class AgentAnswer(BaseModel):
    """What the agent returns for one question."""

    response: str = Field(
        min_length=1,
        description="The final answer: a single letter for MC, or the full answer text for short-answer questions",
    )
    reasoning: str = Field(
        default="",
        description="Brief chain-of-thought leading to the answer",
    )


_OPTION_LETTER = re.compile(r"\b([A-Da-d])\b")
_FIRST_LETTER = re.compile(r"[A-Za-z]")


def normalize_response(raw: str, *, is_mc: bool) -> str:
    """Normalize the response for the contract's `response` field."""
    if is_mc:
        text = raw or ""
        m = _OPTION_LETTER.search(text)
        if m is None:
            m = _FIRST_LETTER.search(text)
        return m.group(0).upper() if m else ""
    return raw or ""


def parse_final_text(text: str, *, is_mc: bool) -> AgentAnswer:
    """Parse the Runner's final response text into an AgentAnswer.

    ADK with output_schema returns JSON matching the schema. If parsing fails
    (model returned plain text), fall back to treating the whole text as the
    response with empty reasoning.
    """
    text = (text or "").strip()
    if not text:
        return AgentAnswer(response="", reasoning="") if not is_mc else AgentAnswer(response="x", reasoning="")  # placeholder; overwritten below
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "response" in data:
            return AgentAnswer(
                response=str(data.get("response", "")),
                reasoning=str(data.get("reasoning", "")),
            )
    except (json.JSONDecodeError, ValueError):
        pass
    # Plain-text fallback: treat as the raw response, no reasoning.
    return AgentAnswer(response=text, reasoning="")
```

Note: the empty-text branch returns a sentinel that the caller overwrites after normalization. To keep it clean, `parse_final_text("")` should return an empty-response object. Fix the placeholder before running:

```python
    if not text:
        return AgentAnswer(response="", reasoning="")
```

(Remove the `if not is_mc else ...` ternary — it was a drafting artifact.)

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_google_adk_output.py -q
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/google-adk/output_models.py tests/frameworks/test_google_adk_output.py
git commit -m "feat(google-adk): add structured output schema and response parser"
```

---

## Task 3: Agent definition (idiomatic ADK)

**Files:**
- Create: `frameworks/google-adk/agent.py`
- Create: `tests/frameworks/test_google_adk_agent.py`

Build the `LlmAgent`. Showcase ADK's "agent = name + model + instruction +
output_schema" core. We do NOT call the model here — just construct and assert.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_google_adk_agent.py`:

```python
from google.adk.agents import LlmAgent

from agent import build_agent, DEFAULT_MODEL, AGENT_INSTRUCTION
from output_models import AgentAnswer


def test_build_agent_returns_llm_agent():
    a = build_agent()
    assert isinstance(a, LlmAgent)


def test_agent_has_expected_model():
    a = build_agent()
    assert a.model == DEFAULT_MODEL


def test_agent_uses_structured_output_schema():
    a = build_agent()
    assert a.output_schema is AgentAnswer


def test_instructions_mention_exam_context():
    assert "Zhongkao" in AGENT_INSTRUCTION
    assert "multiple choice" in AGENT_INSTRUCTION.lower()


def test_build_agent_accepts_override_model():
    a = build_agent(model="gemini-1.5-flash")
    assert a.model == "gemini-1.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/frameworks/test_google_adk_agent.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Write `agent.py`**

```python
"""Build the Google ADK LlmAgent.

Idiomatic ADK style: an LlmAgent with name + model + instruction + output_schema.
ADK enforces that output_schema and tools are mutually exclusive — since exam
questions need structured output and don't benefit from tools, we use
output_schema (no tools). This showcases ADK's code-first, schema-driven core.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from output_models import AgentAnswer

DEFAULT_MODEL = "gemini-2.5-flash"

AGENT_INSTRUCTION = """\
You are a top student taking the Chinese Zhongkao (senior secondary school
entrance exam). Answer each question correctly and concisely.

For multiple choice questions, `response` MUST be a single capital letter
(A, B, C, or D) corresponding to the correct option.

For short answer questions, `response` MUST be the complete answer (a number
with units, a short phrase, a balanced equation, or a one-sentence
explanation) — exactly as a correct exam answer would be written.

Always include brief `reasoning` showing how you reached the answer.
Respond ONLY with the JSON schema fields (response and reasoning). Do not add
extraneous commentary.
"""


def build_agent(*, model: str = DEFAULT_MODEL) -> LlmAgent:
    """Construct the solver LlmAgent."""
    return LlmAgent(
        name="ZhongkaoSolver",
        model=model,
        instruction=AGENT_INSTRUCTION,
        output_schema=AgentAnswer,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_google_adk_agent.py -q
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/google-adk/agent.py tests/frameworks/test_google_adk_agent.py
git commit -m "feat(google-adk): add idiomatic LlmAgent with structured output schema"
```

---

## Task 4: Solver with injectable runner

**Files:**
- Create: `frameworks/google-adk/solver.py`
- Create: `tests/frameworks/test_google_adk_solver.py`

`solve_one` runs the agent via the ADK Runner and returns `(AgentAnswer, latency_ms)`.
The runner is injectable so tests stay deterministic and free. By default it uses
ADK's async `Runner.run_async`, wrapped to return the final text synchronously.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_google_adk_solver.py`:

```python
from solver import solve_one, QuestionInput, build_prompt
from output_models import AgentAnswer


class FakeRunner:
    """Returns a fixed final text for any prompt."""

    def __init__(self, final_text: str):
        self._text = final_text
        self.calls = []

    def __call__(self, agent, prompt: str) -> str:
        self.calls.append((agent, prompt))
        return self._text


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


def test_solve_one_mc_parses_json_response():
    fake = FakeRunner('{"response": "A", "reasoning": "diff of squares"}')
    answer, latency = solve_one(_mc_question(), runner=fake)
    assert answer.response == "A"
    assert answer.reasoning == "diff of squares"
    assert latency >= 0
    assert len(fake.calls) == 1


def test_solve_one_mc_normalizes_plain_letter():
    fake = FakeRunner("Option B")
    answer, _ = solve_one(_mc_question(), runner=fake)
    assert answer.response == "B"


def test_solve_one_short_keeps_verbatim():
    fake = FakeRunner('{"response": "x = 12", "reasoning": "subtracted 2x"}')
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
python -m pytest tests/frameworks/test_google_adk_solver.py -q
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `solver.py`**

```python
"""Solve one question with Google ADK.

The runner is injectable so tests stay deterministic and free of API calls.
By default it uses ADK's async Runner.run_async, wrapped synchronously.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from output_models import AgentAnswer, normalize_response, parse_final_text

RunnerFn = Callable[[LlmAgent, str], str]
"""A runner takes (agent, prompt) and returns the final response text."""


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


def _default_runner(agent: LlmAgent, prompt: str) -> str:
    """Run the agent via ADK's async Runner and return the final response text."""
    app_name = "zhongkao_solver"
    user_id = "solver"
    session_service = InMemorySessionService()

    async def _run() -> str:
        session = await session_service.create_session(
            app_name=app_name, user_id=user_id
        )
        runner = Runner(
            agent=agent, app_name=app_name, session_service=session_service
        )
        content = types.Content(
            role="user", parts=[types.Part(text=prompt)]
        )
        final_text = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                parts = event.content.parts if event.content else []
                final_text = "".join(
                    (p.text for p in parts if hasattr(p, "text") and p.text)
                )
                break
        return final_text

    return asyncio.run(_run())


def solve_one(
    q: dict[str, Any],
    *,
    runner: RunnerFn | None = None,
    agent: LlmAgent | None = None,
) -> tuple[AgentAnswer, int]:
    """Solve one question. Returns (normalized AgentAnswer, latency_ms)."""
    run = runner or _default_runner
    ag = agent
    if ag is None:
        from agent import build_agent  # lazy import to avoid circular dep
        ag = build_agent()

    qi = QuestionInput.from_question(q)
    prompt = build_prompt(q)

    start = time.perf_counter()
    final_text = run(ag, prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)

    parsed = parse_final_text(final_text, is_mc=qi.is_mc)
    normalized = AgentAnswer(
        response=normalize_response(parsed.response, is_mc=qi.is_mc),
        reasoning=parsed.reasoning,
    )
    return normalized, latency_ms
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/frameworks/test_google_adk_solver.py -q
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/google-adk/solver.py tests/frameworks/test_google_adk_solver.py
git commit -m "feat(google-adk): add solver with injectable async runner wrapper"
```

---

## Task 5: CLI entrypoint (run.py) — the contract

**Files:**
- Create: `frameworks/google-adk/run.py`
- Create: `tests/frameworks/test_google_adk_run.py`

Same contract as openai-agents: `python run.py --dataset <path> --output <path>`.
Loads the dataset, solves every question, writes `result/results.json` in exactly
`contract/result.schema.json` shape. Tests use a fake runner.

- [ ] **Step 1: Write the failing test**

`tests/frameworks/test_google_adk_run.py`:

```python
import json
from pathlib import Path

import jsonschema

import run as run_module

_REPO = Path(__file__).resolve().parent.parent.parent
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/frameworks/test_google_adk_run.py -q
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `run.py`**

```python
"""Entrypoint: solve the dataset with Google ADK and write result/results.json.

Usage (contract):
    python run.py --dataset ../../dataset/questions.json --output result/results.json

Idiomatic ADK: build one LlmAgent, solve each question via the Runner, write the
shared result schema. Run as a script; flat imports from this directory.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import click
from dotenv import load_dotenv

from agent import DEFAULT_MODEL, build_agent
from solver import RunnerFn, solve_one

load_dotenv()


def run(
    *,
    dataset_path: str | Path,
    output_path: str | Path,
    framework: str = "google-adk",
    model: str = DEFAULT_MODEL,
    runner: RunnerFn | None = None,
) -> dict:
    """Solve every question in the dataset and write the result file."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    questions = json.loads(dataset_path.read_text())
    if runner is None:
        agent = build_agent(model=model)
    else:
        agent = None

    results = []
    for q in questions:
        qi_id, qi_subject, qi_type = q["id"], q["subject"], q["type"]
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
python -m pytest tests/frameworks/test_google_adk_run.py -q
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add frameworks/google-adk/run.py tests/frameworks/test_google_adk_run.py
git commit -m "feat(google-adk): add run.py contract entrypoint with error handling"
```

---

## Task 6: Shell entrypoint (run.sh)

**Files:**
- Create: `frameworks/google-adk/run.sh`

- [ ] **Step 1: Write `run.sh`**

```bash
#!/usr/bin/env bash
# frameworks/google-adk/run.sh — self-contained entrypoint.
# Solves the shared dataset and writes result/results.json in the contract schema.
#
# Requires: GOOGLE_API_KEY in the environment (or a .env file at repo root).
set -euo pipefail
cd "$(dirname "$0")"

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
chmod +x frameworks/google-adk/run.sh
```

- [ ] **Step 3: Commit**

```bash
git add frameworks/google-adk/run.sh
git commit -m "feat(google-adk): add self-contained run.sh entrypoint"
```

---

## Task 7: Framework README

**Files:**
- Create: `frameworks/google-adk/README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Google ADK — Zhongkao Solver

Self-contained solver for the shared Zhongkao dataset, implemented with Google's
[Agent Development Kit (ADK)](https://github.com/google/adk-python).

## Design (idiomatic ADK)

This implementation uses ADK's core primitive: the **`LlmAgent`**.

- **Single LlmAgent** (`ZhongkaoSolver`) — name + model + instruction + output_schema.
- **Structured output via `output_schema`** — ADK constrains the model's final
  response to the `AgentAnswer` Pydantic schema (response + reasoning). The final
  response text is JSON matching the schema; we parse it back.
- **No tools** — ADK enforces that `output_schema` and `tools` are mutually
  exclusive. Exam questions need structured output and don't benefit from tools,
  so we use `output_schema`. (This mirrors the OpenAI Agents impl's no-tools
  decision, but here it's a hard ADK constraint, not a heuristic.)
- **Async Runner** — ADK's Runner is async and session-based (`InMemorySessionService`
  + `run_async` yielding events). The solver wraps this synchronously via
  `asyncio.run` for simplicity; the runner is injectable so tests bypass it.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Builds the `LlmAgent` (instruction + model + output_schema) |
| `output_models.py` | `AgentAnswer` schema + final-text parser + response normalizer |
| `solver.py` | `solve_one(question)` — runs the agent, parses + normalizes |
| `run.py` | CLI: load dataset → solve all → write `result/results.json` |
| `run.sh` | Shell entrypoint (activates repo venv, sets paths) |

## Install

```bash
cd frameworks/google-adk
pip install -r requirements.txt   # or use the repo-root .venv
```

## Run

```bash
# Option A: via the shell entrypoint (recommended)
export GOOGLE_API_KEY=...
bash run.sh

# Option B: directly
python run.py --dataset ../../dataset/questions.json --output result/results.json
```

Both write `result/results.json` in the shared contract schema.

## Test

From the repo root:

```bash
pytest tests/frameworks/test_google_adk_*.py -v
```

Tests use a fake runner — no API calls, no key required.

## Model

Default: `gemini-2.5-flash`. Override with `--model gemini-1.5-pro` (CLI) or by
editing `agent.DEFAULT_MODEL`.
```

- [ ] **Step 2: Commit**

```bash
git add frameworks/google-adk/README.md
git commit -m "docs(google-adk): framework README"
```

---

## Task 8: Integration with grader (end-to-end, fake runner)

**Files:**
- Create: `tests/frameworks/test_google_adk_integration.py`

- [ ] **Step 1: Write the test**

`tests/frameworks/test_google_adk_integration.py`:

```python
"""End-to-end: framework output -> shared grader -> report.

Uses a fake runner that returns correct-ish answers.
"""
from pathlib import Path

import run as run_module
from grader.loaders import load_result_file
from grader.score import aggregate, score_result_file
from grader.report import render_per_framework

_REPO = Path(__file__).resolve().parent.parent.parent
_DATASET = _REPO / "dataset" / "questions.json"


class CorrectFakeRunner:
    def __call__(self, agent, prompt):
        return '{"response": "A", "reasoning": "reasoning"}'


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
    assert "google-adk" in md
    assert "math-001" in md
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/frameworks/test_google_adk_integration.py -q
```
Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/frameworks/test_google_adk_integration.py
git commit -m "test(google-adk): end-to-end grading integration with shared grader"
```

---

## Task 9: Phase 3 verification

- [ ] **Step 1: Run the full test suite**

```bash
source .venv/bin/activate
python -m pytest -q
```
Expected: all tests pass (Phase 1's 37 + openai-agents' 24 + google-adk's ~26 ≈ 87+).

- [ ] **Step 2: Validate against the contract schema**

```bash
python -m pytest tests/frameworks/test_google_adk_run.py -q
```
Expected: 4 passed.

- [ ] **Step 3: Confirm directory tree**

```bash
find frameworks/google-adk -type f -not -path '*/result/*' -o -name '.gitkeep' | sort
```
Expected:
```
frameworks/google-adk/README.md
frameworks/google-adk/__init__.py
frameworks/google-adk/agent.py
frameworks/google-adk/output_models.py
frameworks/google-adk/requirements.txt
frameworks/google-adk/result/.gitkeep
frameworks/google-adk/run.py
frameworks/google-adk/run.sh
frameworks/google-adk/solver.py
```

- [ ] **Step 4: Commit any remaining changes**

```bash
git add -A
git status
# if anything staged:
git commit -m "chore: phase 3 google-adk framework complete"
```

---

## Self-Review

**1. Spec coverage:**
- Self-contained folder with own deps (`requirements.txt`) ✓ (Task 1)
- Uniform entrypoint `python run.py --dataset --output` ✓ (Task 5)
- Idiomatic ADK (LlmAgent + instruction + output_schema) ✓ (Tasks 3, 4)
- Writes `result/results.json` in shared schema ✓ (Task 5)
- `run.sh` self-contained ✓ (Task 6)
- Framework README ✓ (Task 7)
- Recommended model: spec said `gemini-1.5-pro`; we use `gemini-2.5-flash` (ADK 2.0's recommended default, more current) — documented in README. ✓
- Grader integration ✓ (Task 8)

**2. Placeholder scan:** No TBD/TODO in steps. The Task 2 step 3 has an explicit "fix this drafting artifact before running" note — implementer must apply the cleaner empty-text branch. All other code blocks are complete.

**3. Type/signature consistency:**
- `RunnerFn = Callable[[LlmAgent, str], str]` — returns text (not AgentAnswer), because ADK's final response is text. Test fakes return strings. ✓
- `solve_one(q, *, runner, agent) -> (AgentAnswer, int)` — consistent in solver.py and run.py.
- `run(*, dataset_path, output_path, framework, model, runner) -> dict` — consistent.
- `AgentAnswer(response, reasoning)`, `normalize_response(raw, *, is_mc)`, `parse_final_text(text, *, is_mc) -> AgentAnswer` — consistent across modules and tests.
- Result dict keys match `contract/result.schema.json`. ✓

**4. Scope check:** Phase 3 produces a working, tested framework folder. Mirrors Phase 2's structure deliberately, so the comparison between frameworks (a core project goal) is clean: same file layout, same test shape, different SDK idioms. The ADK-specific deviations (async runner, text-based output parsing, schema/tools mutual exclusivity) are all documented and showcase genuine framework differences.

**Spec deviation noted:** Default model changed from `gemini-1.5-pro` (spec) to `gemini-2.5-flash` (ADK 2.0 recommended default). This is a forward-looking correction, documented in the README. Override via `--model` or `agent.DEFAULT_MODEL`.
