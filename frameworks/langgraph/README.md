# LangGraph — Entrance Exam Solver

Self-contained solver for the shared senior secondary school entrance exam
dataset, implemented with [LangGraph](https://github.com/langchain-ai/langgraph).

## Design (idiomatic LangGraph)

This implementation uses LangGraph's distinctive strength: **explicit,
inspectable control flow as a graph**. The two-loop multi-agent design from
`docs/agents-design.md` is a real graph cycle here, not an imperative loop.

```
START → resolve → validate → [ accepted OR attempts ≥ max ? END : resolve ]
```

- **`State`** (`state.py`) — a `TypedDict` carrying the question, current
  answer, attempts, feedback, and the validator's verdict across nodes.
- **`resolve` node** — produces `{response, reasoning}` via the LLM.
- **`validate` node** — independently checks the answer, returns
  `{accepted, feedback, validator_answer}`.
- **conditional edge** after `validate` (`graph._route_after_validate`) — the
  validate→revise decision: `END` if accepted or `attempts >= MAX_ATTEMPTS`,
  else back to `resolve`. This edge is the idiomatic LangGraph construct that
  the other frameworks simulate imperatively.
- **No tools / no handoffs** — exam questions don't benefit from them; the
  graph's conditional edge is the whole point.

`add_conditional_edges` is given an explicit `path_map`
(`{"resolve": "resolve", END: END}`) to avoid LangGraph
[issue #987](https://github.com/langchain-ai/langgraph/issues/987) (without
the map, the graph would add edges to every node).

## Model

LangGraph is model-agnostic. The default is `gpt-4o` via `langchain-openai`'s
`ChatOpenAI` (set `OPENAI_API_KEY`). Override with `--model`, or pass a custom
`LLM` (anything with `.invoke(prompt) -> str`) to `solve_one` / `run`.

## Files

| File | Purpose |
|---|---|
| `state.py` | The graph `State` TypedDict |
| `graph.py` | `build_graph()` — resolve/validate nodes + conditional edge (the core) |
| `llm.py` | `build_llm()` — default `ChatOpenAI` wrapper; injectable |
| `output_models.py` | `AgentAnswer` / `ValidatorVerdict` / `QuestionStatus` + parsers |
| `solver.py` | `solve_one(question)` — compile + invoke the graph |
| `run.py` | CLI: outer loop over dataset → write `result/results.json` |
| `run.sh` | Shell entrypoint (activates repo venv, sets paths) |

## Install

```bash
cd frameworks/langgraph
pip install -r requirements.txt   # or use the repo-root .venv
```

## Run

```bash
# Option A: via the shell entrypoint (recommended)
export OPENAI_API_KEY=sk-...
bash run.sh

# Option B: directly
python run.py --dataset ../../dataset/questions.json --output result/results.json

# Disable the Validator (single-agent baseline):
python run.py --dataset ../../dataset/questions.json --output result/results.json --no-validator
```

All write `result/results.json` in the shared contract schema.

## Test

From the repo root:

```bash
pytest tests/frameworks/test_langgraph_*.py -v
```

Tests inject a fake `LLM` — no API calls, no key required. The graph tests
(`test_langgraph_graph.py`) verify the conditional-edge routing directly:
accept-first-try, revise-then-accept, cap-at-max, and validator-disabled.
