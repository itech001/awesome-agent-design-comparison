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
