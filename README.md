# Awesome Agent Design Comparison

A benchmark and design-comparison project that evaluates how different LLM agent
frameworks solve the same exam task — the senior secondary school entrance exam.

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

50 senior secondary school entrance exam questions, translated to English, covering:
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

MIT (code). Dataset questions are based on/inspired by real past senior secondary
school entrance exam
papers, translated to English; see `source` fields for attribution.
