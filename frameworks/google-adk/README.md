# Google ADK — Entrance Exam Solver

Self-contained solver for the shared senior secondary school entrance exam
dataset, implemented with Google's
[Agent Development Kit (ADK)](https://github.com/google/adk-python).

## Design (idiomatic ADK)

This implementation uses ADK's core primitive: the **`LlmAgent`**.

- **Single LlmAgent** (`ExamSolver`) — name + model + instruction + output_schema.
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
