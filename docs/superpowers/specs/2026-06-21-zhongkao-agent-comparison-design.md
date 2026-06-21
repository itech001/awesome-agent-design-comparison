# Zhongkao Agent Comparison — Design Spec

**Date:** 2026-06-21
**Status:** Approved (pending implementation)
**Owner:** project maintainer

## 1. Purpose

A benchmark and design-comparison project that evaluates how different LLM agent
frameworks solve the same exam task — the Chinese **Zhongkao** (中考, the senior
secondary school entrance exam).

The project serves four goals simultaneously:

1. **Benchmark accuracy** — which framework answers most correctly?
2. **Compare design patterns** — how does each framework structure the same task?
3. **Showcase** — idiomatic reference implementations you can study.
4. **Reproducibility** — one dataset, one grader, one report.

## 2. Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary goal | All four (benchmark + design + showcase + accuracy) | Comprehensive comparison |
| Frameworks | OpenAI Agents SDK, Google ADK, LangGraph, CrewAI | Four popular, stylistically distinct frameworks |
| Subjects | Chinese, Math, English, Physics, Chemistry, Biology, History, Geography | Core 3 + 3 sciences + 2 humanities |
| Question types | Multiple choice + short answer | Auto-gradeable spectrum |
| Grading | Hybrid (deterministic rubric default, optional LLM-judge) | Reproducible by default, nuanced on demand |
| Question language | All English | English-default project; translatable later |
| Architecture | Shared dataset + shared result contract; each framework self-contained & idiomatic | Comparable outputs, authentic code |
| Output location | `frameworks/<name>/result/` (per-framework, self-contained) | Consistent with "fully self-contained" principle |
| Orchestration | `run_all.sh` full pipeline, `--skip-grade` to skip | One command = full benchmark |

## 3. Repository Layout

```
awesome-agent-design-comparison/
├── README.md                      # English project intro
├── run_all.sh                     # run all frameworks + grade + report (--skip-grade)
├── dataset/
│   ├── questions.json             # the 50 questions — ONE shared copy
│   └── schema.md                  # question schema doc
├── contract/                      # shared agreement between dataset/frameworks/grader
│   ├── question.schema.json       # input schema (JSON Schema)
│   ├── result.schema.json         # output schema every framework must write
│   └── README.md
├── frameworks/                    # each fully self-contained & idiomatic
│   ├── openai-agents/
│   │   ├── run.sh                 # activate venv, set env, call run.py
│   │   ├── run.py                 # entrypoint: --dataset, --output
│   │   ├── requirements.txt       # own deps
│   │   ├── README.md              # how to install + run
│   │   └── result/                # gitignored outputs (result/results.json)
│   ├── google-adk/
│   │   └── result/
│   ├── langgraph/
│   │   └── result/
│   └── crewai/
│       └── result/
├── grader/                        # shared hybrid grader, framework-agnostic
│   ├── grade.py                   # CLI: score a single result file
│   ├── grade_all.py               # scores every frameworks/*/result/results.json
│   ├── report.py                  # writes comparison report
│   ├── graders/
│   │   ├── exact.py               # MC: exact letter match
│   │   ├── rubric.py              # short_answer: deterministic rubric scorer
│   │   └── llm_judge.py           # optional LLM-as-judge scorer
│   └── requirements.txt
├── reports/                       # generated reports (gitignored)
└── docs/                          # design docs, per-framework notes
```

### Contract rules

- **Frameworks depend on `dataset/` and `contract/` (read-only).** They never copy
  or redefine the schema.
- **Frameworks write to `frameworks/<name>/result/results.json`** using exactly
  `contract/result.schema.json`. Top-level field names and types are fixed; a
  `raw` field lets each framework stash framework-specific detail (traces, tool
  calls, agent state) without breaking comparability.
- **All four frameworks are Python.** OpenAI Agents SDK, Google ADK, LangGraph,
  and CrewAI all have first-class Python SDKs. Each framework folder owns its own
  `requirements.txt` and recommended venv, so dependency conflicts are isolated.
- **Each framework has a uniform entrypoint** — `python run.py --dataset <path>
  --output result/results.json` — but the internals (agents, tools, orchestration)
  are 100% idiomatic to that framework.

## 4. Dataset

**50 questions, 8 subjects, all in English**, distributed as:

| Subject | MC | Short answer | Total |
|---|---|---|---|
| Chinese (语文) | 5 | 3 | 8 |
| Math (数学) | 4 | 4 | 8 |
| English (英语) | 5 | 3 | 8 |
| Physics (物理) | 4 | 2 | 6 |
| Chemistry (化学) | 4 | 2 | 6 |
| Biology (生物) | 4 | 2 | 6 |
| History (历史) | 3 | 1 | 4 |
| Geography (地理) | 3 | 1 | 4 |
| **Total** | **32** | **18** | **50** |

The three core subjects sit at 8; the three lab sciences at 6; the two humanities
at 4 (mostly multiple choice, one short-answer each). Questions are based on real
past Zhongkao papers, translated to English.

### Question schema (`dataset/questions.json`)

```jsonc
[
  {
    "id": "math-001",                     // <subject>-<seq>, globally unique
    "subject": "math",                    // chinese|math|english|physics|chemistry|biology|history|geography
    "type": "multiple_choice",            // multiple_choice | short_answer
    "question": "Simplify: (x+2)(x-2)",   // English prompt
    "options": [                          // MC only; short_answer omits this
      "A. x²-4", "B. x²+4", "C. x²-2x", "D. x²-4x+4"
    ],
    "answer": "A",                        // MC: letter; short_answer: reference string
    "answer_type": "letter",              // letter | text  (drives grader behavior)
    "scoring": {
      "max_points": 1,                    // MC=1; short_answer varies 2-5
      "method": "exact",                  // exact | rubric
      "rubric": [                         // present only when method=rubric
        {"criterion": "sets up equation", "points": 1},
        {"criterion": "correct computation", "points": 1},
        {"criterion": "correct final answer", "points": 1}
      ]
    },
    "source": "2023 Beijing Zhongkao Q15", // attribution for authenticity
    "difficulty": "medium",               // easy | medium | hard
    "notes": ""                           // optional grader hints, e.g. accepted variants
  }
]
```

Design points:
- **`answer_type`** drives grading: `letter` → exact match; `text` → rubric.
- **`scoring`** lives *with* the question, so the grader follows the method the
  question specifies — the grader itself stays simple and reproducible.
- **`source`** keeps the dataset authentic and verifiable.
- **`notes`** is the escape hatch for accepted-answer variants (units, equivalent
  forms) without overcomplicating the rubric.

## 5. Result Schema

Every framework writes `frameworks/<name>/result/results.json` in this exact shape
(`contract/result.schema.json`):

```jsonc
{
  "framework": "openai-agents",           // matches the folder name
  "model": "gpt-4o",                      // model used
  "generated_at": "2026-06-21T10:30:00Z", // ISO 8601 UTC
  "config": {                             // framework-specific, free-form
    "temperature": 0.0,
    "runs_per_question": 1
  },
  "results": [
    {
      "question_id": "math-001",
      "subject": "math",
      "type": "multiple_choice",
      "response": "A",                    // the agent's final answer
      "reasoning": "...",                 // chain-of-thought, may be empty
      "latency_ms": 1234,                 // wall-clock for this question
      "raw": { }                          // framework-specific detail (traces, tool calls)
    }
  ]
}
```

Design points:
- Fixed top-level fields make every framework's output directly comparable.
- `raw` is the safety valve: anything framework-native (CrewAI task outputs,
  LangGraph state dumps, ADK events) goes here without breaking comparability.
- `reasoning` is separate from `response` so we grade the final answer but can
  also analyze reasoning quality later.

## 6. Grader

Shared, framework-agnostic. Scores any `result/results.json` against the dataset.

### Grading logic

- **Multiple choice** (`answer_type: letter`) → exact match. `correct=1` or `0`.
- **Short answer** (`answer_type: text`, `method: rubric`) → rubric scorer checks
  each criterion. Two modes:
  - **Default (deterministic):** keyword/regex/string-similarity check per
    criterion against `question.notes` hints. Fully reproducible.
  - **Optional (`--judge llm`):** LLM-as-judge scores each criterion 0/full.
    Cost-bearing, non-deterministic; off by default.

Scores are **normalized to `[0,1]` per question** (earned / max_points), then
averaged. This lets MC (1pt) and short-answer (3pt) contribute comparably across
subjects and frameworks.

### Report output (`reports/`)

- `reports/<framework>-<timestamp>.md` — per-framework breakdown: overall
  accuracy, per-subject accuracy, per-type accuracy, latency stats.
- `reports/comparison-<timestamp>.md` — cross-framework table: each framework's
  overall + per-subject scores side by side, plus a **design-pattern analysis**
  section (LOC, file count, dependency count, pattern type per framework — the
  "design comparison" goal).

### CLI

```
python grader/grade.py <result.json> --dataset <questions.json> [--judge llm]
python grader/grade_all.py               # scores every frameworks/*/result/results.json
python grader/report.py                  # writes reports/comparison-<ts>.md
```

## 7. Per-Framework Implementation

Each framework folder is fully self-contained and idiomatic, but all conform to
the same contract.

### Common structure (the contract part)

```
frameworks/<name>/
├── run.sh                # entrypoint: cd to own dir, activate venv, call run.py
├── run.py                # python entrypoint: --dataset, --output result/results.json
├── requirements.txt      # own deps, isolated venv recommended
├── README.md             # how to install + run, what's idiomatic about this impl
└── result/               # gitignored outputs
```

### Per-framework `run.sh` contract

Self-contained, knows nothing about siblings:

```bash
#!/usr/bin/env bash
# frameworks/openai-agents/run.sh
set -euo pipefail
cd "$(dirname "$0")"
# optional venv (documented in README.md, not forced):
#   python -m venv .venv && source .venv/bin/activate
python run.py --dataset ../../dataset/questions.json --output result/results.json
```

- Portable on its own — `bash frameworks/crewai/run.sh` works standalone.
- Paths resolve relative to each script's own location, so it works from any cwd.

### Idiomatic approach per framework

**1. `openai-agents/` (OpenAI Agents SDK)**
- Define `Agent` objects with `instructions`, optional `tools`, use the SDK's
  `Runner.run()` loop. Leverage native tool-calling and structured output
  (`output_type` Pydantic model) for clean `response`/`reasoning`.
- Style: thin, async, explicit tool definitions. Showcases the SDK's
  "agent = instructions + tools + model" minimalism.

**2. `google-adk/` (Google ADK)**
- Define an `LlmAgent` with `instruction`, `tools`, possibly a `SequentialAgent`
  or sub-agents per subject (math agent, science agent, humanities agent) to
  showcase ADK's multi-agent orchestration strengths.
- Style: code-first, model-agnostic. Showcases hierarchical composition and
  ADK's session/state model.

**3. `langgraph/` (LangGraph)**
- Build a `StateGraph` — e.g. route-by-subject node → solve node → answer-format
  node. Use tools and conditional edges. Typed `State` with Pydantic.
- Style: explicit graph definition, checkpoints. Showcases controllable,
  inspectable, graph-based workflows.

**4. `crewai/` (CrewAI)**
- Define `Agent` roles (e.g. "Math Expert", "Science Expert", "Humanities
  Expert", plus a "Reviewer") and a `Crew` with `Tasks`. Subject-based routing
  via role assignment.
- Style: role-playing, task-delegating collaboration. Showcases the
  collaborative-agent/team paradigm.

### Design-comparison metrics (auto-captured for the report)

- Lines of code, file count, dependency count (from `requirements.txt`).
- Whether the framework uses tools, multi-agent, graph, or role-based patterns.
- A short "developer ergonomics" note per framework (authored, not auto-generated).

These metrics feed the `reports/comparison-*.md` design-analysis section.

## 8. Runner Orchestration

`run_all.sh` at the repo root runs the full pipeline:

```bash
#!/usr/bin/env bash
# run_all.sh — run all frameworks, grade, and generate report.
# Usage:
#   ./run_all.sh                 # run all + grade + report
#   ./run_all.sh --skip-grade    # run all frameworks only
set -euo pipefail
cd "$(dirname "$0")"

SKIP_GRADE=0
[[ "${1:-}" == "--skip-grade" ]] && SKIP_GRADE=1

for fw in frameworks/*/; do
  name=$(basename "$fw")
  echo "=== Running $name ==="
  bash "$fw/run.sh"
done

if [[ "$SKIP_GRADE" -eq 0 ]]; then
  echo "=== Grading all ==="
  python grader/grade_all.py
  echo "=== Generating report ==="
  python grader/report.py
else
  echo "Skipping grade + report (--skip-grade)"
fi
```

- Defaults to the full pipeline (one command = full benchmark + report).
- `--skip-grade` lets you iterate on a framework's code without waiting for
  grading.
- Uses a glob (`frameworks/*/`), so adding a 5th framework later needs zero edits
  to the orchestrator — just drop in a new folder with its own `run.sh`.

## 9. README Content

The README is the English project intro (default English, translatable later).
Structure:

```markdown
# Awesome Agent Design Comparison

A benchmark and design-comparison project that evaluates how different LLM agent
frameworks solve the same exam task — the Chinese Zhongkao (中考, the senior
secondary school entrance exam).

We build a shared dataset of 50 translated exam questions spanning 8 subjects,
then implement the same solver independently with four popular agent frameworks.
A shared grader scores every implementation against the same rubric, so the
results are directly comparable.

## Why
- Benchmark accuracy: which framework answers most correctly?
- Compare design patterns: how does each framework structure the same task?
- Showcase: idiomatic reference implementations you can study.
- Reproducibility: one dataset, one grader, one report.

## The Dataset
50 Zhongkao questions, translated to English, covering:
Chinese, Math, English, Physics, Chemistry, Biology, History, Geography.
Question types: multiple choice (32) + short answer (18). See `dataset/`.

## Frameworks Compared
| Framework | Style | Folder |
|---|---|---|
| OpenAI Agents SDK | minimal agent + tools, async | `frameworks/openai-agents/` |
| Google ADK | hierarchical multi-agent orchestration | `frameworks/google-adk/` |
| LangGraph | explicit state-graph workflows | `frameworks/langgraph/` |
| CrewAI | role-based collaborative crews | `frameworks/crewai/` |

Each is fully self-contained — its own deps, venv, and `run.sh`.

## Quick Start
1. Set API keys (e.g. `export OPENAI_API_KEY=...`, `export GOOGLE_API_KEY=...`).
2. `./run_all.sh` — runs all frameworks, grades, and writes a comparison report.
3. See `reports/` for results.

Run one framework: `bash frameworks/openai-agents/run.sh`.

## How It Works
Shared dataset → each framework writes result/results.json in a fixed schema →
shared grader scores → report.

## Project Layout
[tree from Section 3]

## Grading
Hybrid: deterministic rubric by default (reproducible), optional LLM-as-judge
mode (`--judge llm`). Scores normalized to [0,1] per question.

## Extending
Add a framework: drop a new `frameworks/<name>/` with `run.sh` + `run.py` that
honors `contract/result.schema.json`. `run_all.sh` picks it up automatically.

## License & Attribution
[dataset sourced from past Zhongkao papers, translated; license TBD]
```

## 10. Out of Scope (YAGNI)

The following are intentionally **not** in this spec, to keep scope tight:

- A web UI / dashboard for results.
- Automated cost tracking (token spend) — `raw` may contain it, but no analysis.
- Multi-model matrix per framework (one recommended model per framework for v1).
- Chinese / other-language dataset versions (future translation effort).
- CI/CD pipeline.
- Statistical significance testing across runs.
- A 5th+ framework (drop-in architecture supports it, but v1 ships four).

## 11. Resolved Items

1. **Dataset authoring** — all 50 questions drafted from real past Zhongkao
   papers during implementation (no user-provided seed).
2. **Recommended model per framework** — decided:
   - OpenAI Agents SDK → `gpt-4o`
   - Google ADK → `gemini-1.5-pro`
   - LangGraph → `gpt-4o` (model-agnostic by default; uses OpenAI for v1)
   - CrewAI → `gpt-4o`
3. **License** — MIT for code, with dataset attribution noted.
