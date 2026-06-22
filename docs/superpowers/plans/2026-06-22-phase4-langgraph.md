# Phase 4: LangGraph Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a fully self-contained `frameworks/langgraph/` folder that solves the shared dataset with LangGraph's `StateGraph`, expressing the two-loop multi-agent design (docs/agents-design.md) as an explicit graph: a `resolve` node, a `validate` node, and a **conditional edge** that loops back to `resolve` on `REVISE` or terminates on `ACCEPT`/max-attempts. Writes `result/results.json` in the shared contract; all tested.

**Architecture:** LangGraph's distinctive strength is *explicit, inspectable control flow as a graph*. The inner loop becomes a real graph cycle:
```
START → resolve → validate → [ accepted OR attempts>=max ? END : resolve ]
```
A `State` TypedDict carries the question, current answer, attempts, feedback, and the validator's verdict across nodes. Each node is a plain Python function over `State`. The conditional edge (`add_conditional_edges`) is the idiomatic LangGraph way to express the validate→revise decision — the exact construct the other frameworks simulate imperatively. The outer loop (walk every question) lives in `run.py`, invoking the compiled graph once per question. The LLM is wrapped behind an injectable callable so tests run without API calls.

**Tech Stack:** Python 3.11+, `langgraph` + `langchain-openai` (model-agnostic; defaults to `gpt-4o` per spec), `pydantic` (structured parsing of node outputs), `pytest`. Reads `dataset/` and `contract/` (read-only); writes `result/`.

**Prerequisite:** Phases 1–3 merged. Branch from `main` (or the phase3 branch base).

---

## File Structure

| File | Responsibility |
|---|---|
| `frameworks/langgraph/requirements.txt` | `langgraph`, `langchain-openai`, `pydantic` |
| `frameworks/langgraph/__init__.py` | Package marker |
| `frameworks/langgraph/output_models.py` | `AgentAnswer`, `ValidatorVerdict`, `QuestionStatus`, `MAX_ATTEMPTS`, normalize/parse helpers |
| `frameworks/langgraph/llm.py` | Build the LLM (`ChatOpenAI`); injectable call interface |
| `frameworks/langgraph/state.py` | The `State` TypedDict + node-bound helpers |
| `frameworks/langgraph/graph.py` | Build the `StateGraph` (resolve/validate nodes + conditional edge); the idiomatic core |
| `frameworks/langgraph/solver.py` | `solve_one(question)` — compile+invoke the graph, return `(AgentAnswer, latency_ms)` |
| `frameworks/langgraph/run.py` | CLI entrypoint: outer loop over dataset → write `result/results.json` |
| `frameworks/langgraph/run.sh` | Self-contained shell entrypoint |
| `frameworks/langgraph/README.md` | How to install/run; what's idiomatic |
| `frameworks/langgraph/result/.gitkeep` | Keeps output dir |

Test files (under `tests/frameworks/langgraph/`, with its own conftest per the cross-framework isolation pattern):
| Test file | Covers |
|---|---|
| `test_langgraph_output.py` | output models + normalize/parse |
| `test_langgraph_graph.py` | graph construction + conditional-edge routing (fake LLM) |
| `test_langgraph_solver.py` | solve_one accept/revise/cap/disable (fake LLM) |
| `test_langgraph_run.py` | end-to-end run.py → contract schema |
| `test_langgraph_integration.py` | framework output → shared grader |

**Import conventions:** Folder `langgraph` has no hyphen — it *is* a valid Python package name. Unlike openai-agents/google-adk, we CAN use a real package. But to stay consistent with the established per-framework test-isolation pattern (each framework's tests live in `tests/frameworks/<name>/` with a conftest that puts that framework's dir on sys.path and clears flat modules), we use flat imports here too. This keeps all four frameworks uniform and avoids a special case.

---

## Task 1: Bootstrap framework folder + import shim

**Files:**
- Create: `frameworks/langgraph/requirements.txt`
- Create: `frameworks/langgraph/__init__.py`
- Create: `frameworks/langgraph/result/.gitkeep`
- Create: `tests/frameworks/langgraph/conftest.py`

- [ ] **Step 1: Create `requirements.txt`**

```
langgraph>=0.2
langchain-openai>=0.2
pydantic>=2.6
```

- [ ] **Step 2: `__init__.py`**

```python
"""LangGraph framework implementation.

Self-contained solver for the senior secondary school entrance exam dataset
using LangGraph. Idiomatic style: the two-loop resolve->validate design is an
explicit StateGraph with a conditional edge (the validate->revise cycle).
"""
```

- [ ] **Step 3: `result/.gitkeep`** (empty)

- [ ] **Step 4: `tests/frameworks/langgraph/conftest.py`** (same isolation pattern as the other frameworks)

```python
"""Make `frameworks/langgraph/` importable for these tests.

LangGraph's folder has no hyphen, but we keep flat imports for consistency
with the other frameworks' test-isolation pattern.
"""
import sys
from pathlib import Path

_FW_DIR = Path(__file__).resolve().parents[3] / "frameworks" / "langgraph"
_FLAT_MODULES = ("output_models", "llm", "state", "graph", "solver", "run")
for _mod in list(sys.modules):
    if _mod in _FLAT_MODULES:
        del sys.modules[_mod]
sys.path.insert(0, str(_FW_DIR))
```

- [ ] **Step 5: Install + verify imports**

```bash
source .venv/bin/activate
pip install -r frameworks/langgraph/requirements.txt
python -c "from langgraph.graph import StateGraph, START, END; print('langgraph OK')"
```
Expected: `langgraph OK`.

- [ ] **Step 6: Commit**

```bash
git add frameworks/langgraph/requirements.txt frameworks/langgraph/__init__.py frameworks/langgraph/result/.gitkeep tests/frameworks/langgraph/conftest.py
git commit -m "feat(langgraph): bootstrap framework folder with import shim"
```

---

## Task 2: Structured output models

**Files:**
- Create: `frameworks/langgraph/output_models.py`
- Create: `tests/frameworks/langgraph/test_langgraph_output.py`

Mirror the other frameworks' models so results are comparable. LangGraph nodes return dicts that update State; the models here parse the LLM's text output within nodes.

- [ ] **Step 1: Failing test** (`test_langgraph_output.py`) — covers `AgentAnswer`, `ValidatorVerdict`, `normalize_response`, `parse_final_text`, `parse_validator_text`, `MAX_ATTEMPTS`. Same shape as google-adk's output test (9 tests).

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Write `output_models.py`** — identical to google-adk's (AgentAnswer with min_length=1; ResolverOutput alias; ValidatorVerdict; QuestionStatus; MAX_ATTEMPTS=3; normalize_response with A-D option letter; parse_final_text raising on empty; parse_validator_text with plain-text fallback).

- [ ] **Step 4: Run → 9 pass.**

- [ ] **Step 5: Commit** `feat(langgraph): add structured output models and parsers`.

---

## Task 3: LLM wrapper

**Files:**
- Create: `frameworks/langgraph/llm.py`
- Create: `tests/frameworks/langgraph/test_langgraph_llm.py`

A thin wrapper around `ChatOpenAI` so nodes can call `llm.invoke(prompt) -> str` and tests can inject a fake. The default model is `gpt-4o` (model-agnostic; langgraph doesn't ship a model).

- [ ] **Step 1: Failing test** — `build_llm()` returns an object with an `invoke(prompt) -> str` method; `build_llm(model="gpt-4o-mini")` overrides; a fake `FakeLLM` with `.invoke` works as a drop-in.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write `llm.py`**

```python
"""LLM wrapper for the LangGraph nodes.

LangGraph is model-agnostic; we use langchain-openai's ChatOpenAI by default.
The node functions call llm.invoke(prompt) -> str, so tests can inject any
object with a compatible .invoke.
"""
from __future__ import annotations
from typing import Protocol

DEFAULT_MODEL = "gpt-4o"


class LLM(Protocol):
    def invoke(self, prompt: str) -> str: ...


def build_llm(*, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> LLM:
    """Construct the default ChatOpenAI LLM. Requires OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature)
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit** `feat(langgraph): add injectable LLM wrapper`.

---

## Task 4: State + graph (the idiomatic core)

**Files:**
- Create: `frameworks/langgraph/state.py`
- Create: `frameworks/langgraph/graph.py`
- Create: `tests/frameworks/langgraph/test_langgraph_graph.py`

This is the showcase: the two-loop design as an explicit graph.

- [ ] **Step 1: Failing test** (`test_langgraph_graph.py`):

```python
from graph import build_graph, State
from output_models import MAX_ATTEMPTS


class FakeLLM:
    """Returns scripted text per call, cycling through a queue per role."""
    def __init__(self, resolve_texts, validate_texts):
        self._r = list(resolve_texts); self._v = list(validate_texts)
    def invoke_resolve(self, p): return self._r.pop(0)
    def invoke_validate(self, p): return self._v.pop(0)


def test_graph_accepts_on_first_try():
    g = build_graph(llm=..., ...)  # see plan detail
    ...
```

(Detail: the test injects a fake that the resolve/validate nodes call. Assert the graph terminates at END with `accepted=True`, `attempts=1`.)

Cover: accept-first-try, revise-then-accept, cap-at-max, disable-validator.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write `state.py`**

```python
"""The graph State. Carries everything across resolve/validate nodes."""
from __future__ import annotations
from typing import TypedDict


class State(TypedDict, total=False):
    question: dict          # the dataset question
    is_mc: bool
    # resolve outputs
    response: str           # current resolver answer (normalized)
    reasoning: str
    raw_response: str       # pre-normalization, for the validator prompt
    # loop bookkeeping
    attempts: int
    feedback: str           # validator's revision note
    accepted: bool
    validator_answer: str
    # control
    enable_validator: bool
    max_attempts: int
```

- [ ] **Step 4: Write `graph.py`** — the idiomatic core:

```python
"""Build the StateGraph: resolve -> validate -> (conditional) resolve | END.

This is LangGraph's idiomatic expression of the two-loop design. The
conditional edge after `validate` is the validate->revise decision.
"""
from __future__ import annotations
from typing import Callable
from langgraph.graph import START, END, StateGraph

from output_models import MAX_ATTEMPTS, normalize_response, parse_final_text, parse_validator_text
from state import State


def _make_resolve_node(invoke_resolve: Callable[[str], str]):
    def resolve(state: State) -> dict:
        q = state["question"]
        prompt = _build_resolver_prompt(q, state.get("feedback", ""))
        raw_text = invoke_resolve(prompt)
        parsed = parse_final_text(raw_text, is_mc=state["is_mc"])
        return {
            "raw_response": parsed.response,
            "response": normalize_response(parsed.response, is_mc=state["is_mc"]),
            "reasoning": parsed.reasoning,
            "attempts": state.get("attempts", 0) + 1,
        }
    return resolve


def _make_validate_node(invoke_validate: Callable[[str], str]):
    def validate(state: State) -> dict:
        raw_text = invoke_validate(_build_validator_prompt(state["question"], state["raw_response"]))
        verdict = parse_validator_text(raw_text, is_mc=state["is_mc"])
        return {
            "accepted": verdict.accepted,
            "feedback": verdict.feedback,
            "validator_answer": normalize_response(verdict.checked_answer, is_mc=state["is_mc"]),
        }
    return validate


def _route_after_validate(state: State) -> str:
    """Conditional edge: END if accepted or capped, else back to resolve."""
    if state.get("accepted") or state.get("attempts", 0) >= state.get("max_attempts", MAX_ATTEMPTS):
        return END
    return "resolve"


def build_graph(*, invoke_resolve, invoke_validate, enable_validator: bool = True):
    builder = StateGraph(State)
    builder.add_node("resolve", _make_resolve_node(invoke_resolve))
    if enable_validator:
        builder.add_node("validate", _make_validate_node(invoke_validate))
        builder.add_edge(START, "resolve")
        builder.add_edge("resolve", "validate")
        builder.add_conditional_edges("validate", _route_after_validate, {"resolve": "resolve", END: END})
    else:
        builder.add_edge(START, "resolve")
        builder.add_edge("resolve", END)
    return builder.compile()
```

(`_build_resolver_prompt` / `_build_validator_prompt` mirror the other frameworks; include them in graph.py or a shared prompts module.)

- [ ] **Step 5: Run → pass** (graph routes correctly with fakes).

- [ ] **Step 6: Commit** `feat(langgraph): add StateGraph with resolve/validate conditional edge`.

---

## Task 5: Solver

**Files:**
- Create: `frameworks/langgraph/solver.py`
- Create: `tests/frameworks/langgraph/test_langgraph_solver.py`

`solve_one(q, *, llm=None, enable_validator=True, max_attempts=MAX_ATTEMPTS) -> (AgentAnswer, int)`. Compiles the graph (or accepts a pre-built one for tests), invokes it with the initial State, maps the final State to `AgentAnswer` + `QuestionStatus` (attached via `__status__`, matching the other frameworks).

- [ ] **Step 1: Failing test** — accept/revise/cap/disable, using an injected fake LLM whose `.invoke` dispatches resolve vs validate based on prompt content (or two separate callables).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write `solver.py`** — builds invoke_resolve/invoke_validate from the llm (default `build_llm()`), builds the graph, invokes `graph.invoke(initial_state)`, constructs `AgentAnswer` + `QuestionStatus` from the final state, attaches `__status__`.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit** `feat(langgraph): add solver wrapping the compiled graph`.

---

## Task 6: run.py — the contract + outer loop

**Files:**
- Create: `frameworks/langgraph/run.py`
- Create: `tests/frameworks/langgraph/test_langgraph_run.py`

Outer loop walks the dataset (coverage guaranteed), invokes `solve_one` per question, writes `result/results.json` validating against `contract/result.schema.json`. `raw` carries `attempts/accepted/validator_answer`.

- [ ] **Step 1: Failing test** — same 4 tests as the other frameworks (writes valid schema, preserves subject/type, records latency, handles errors). Plus a validator-status test.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Write `run.py`** — mirrors openai-agents/google-adk run.py: `run(*, dataset_path, output_path, framework="langgraph", model=DEFAULT_MODEL, llm=None, enable_validator=True) -> dict`; Click CLI with `--no-validator`.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit** `feat(langgraph): add run.py contract entrypoint with outer loop`.

---

## Task 7: run.sh + README

**Files:**
- Create: `frameworks/langgraph/run.sh`
- Create: `frameworks/langgraph/README.md`

- [ ] **Step 1: `run.sh`** — same shape as the other frameworks (cd to own dir, source repo venv, call `python run.py --dataset ../../dataset/questions.json --output result/results.json`).

- [ ] **Step 2: `README.md`** — emphasize the idiomatic difference: **the two-loop design is a real graph cycle here**, not an imperative loop. Document the State, the nodes, the conditional edge. Note model-agnosticism (default gpt-4o via langchain-openai).

- [ ] **Step 3: Commit** `feat(langgraph): add run.sh and framework README`.

---

## Task 8: Integration with grader

**Files:**
- Create: `tests/frameworks/langgraph/test_langgraph_integration.py`

- [ ] **Step 1: Test** — run.py with a fake LLM → load_result_file → score_result_file → aggregate → render_per_framework. Assert count==50, overall in [0,1], report mentions "langgraph" and "math-001".

- [ ] **Step 2: Run → pass.**

- [ ] **Step 3: Commit** `test(langgraph): end-to-end grading integration`.

---

## Task 9: Phase 4 verification

- [ ] **Step 1: Full suite** — `pytest -q` → expect ~130+ tests (Phase 1: 37 + openai-agents: 31 + google-adk: 31 + langgraph: ~31).
- [ ] **Step 2: Contract validation** — `pytest tests/frameworks/langgraph/test_langgraph_run.py -q` → 5 pass.
- [ ] **Step 3: Tree** — `find frameworks/langgraph -type f ...` → all 9 files present.
- [ ] **Step 4: Commit** any remaining; mark phase complete.

---

## Self-Review

**1. Spec coverage:** self-contained folder ✓, uniform entrypoint ✓, idiomatic LangGraph (StateGraph + conditional edge = the two-loop) ✓, shared result schema ✓, run.sh ✓, README ✓, model `gpt-4o` (model-agnostic default) ✓, grader integration ✓.

**2. Placeholder scan:** Task 2/3 test bodies say "same shape as google-adk's" — the implementer should write the actual test code, not reference another file. (Flagged here so it's not skipped.)

**3. Type/signature consistency:** `State` keys (`response`, `reasoning`, `attempts`, `feedback`, `accepted`, `validator_answer`) match `QuestionStatus` and the contract's `raw`. `LLM.invoke(prompt) -> str` is the single seam; fakes conform via duck typing. `solve_one -> (AgentAnswer, int)` matches the other frameworks. `build_graph` returns a compiled graph whose `.invoke(state) -> state` is the LangGraph convention.

**4. Scope:** Phase 4 produces a working, tested framework that showcases LangGraph's distinctive strength (explicit conditional-edge control flow for the validate→revise cycle). No out-of-scope work. The `path_map` argument on `add_conditional_edges` is included to avoid the known issue #987.
